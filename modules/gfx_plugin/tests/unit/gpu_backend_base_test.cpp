// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "backends/metal/compiler/metal_kernel_artifacts.hpp"
#include "backends/metal/compiler/metal_operation_support.hpp"
#include "backends/metal/runtime/stage_factory.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "compiler/backend_registry.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/gfx_compiler_service.hpp"
#include "compiler/manifest.hpp"
#include "compiler/operation_support.hpp"
#include "kernel_ir/gfx_kernel_source.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "kernel_ir/opencl_kernels/binary_f32_kernel.hpp"
#include "kernel_ir/opencl_kernels/interpolate_f32_kernel.hpp"
#include "kernel_ir/opencl_kernels/softmax_f32_kernel.hpp"
#include "kernel_ir/metal_kernels/mpsrt_image_bridge_kernels.hpp"
#include "kernel_ir/metal_kernels/mpsrt_topk_kernels.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/tile.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gpu_backend_base.hpp"
#include "transforms/gfx_llm_ops.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

struct FakeBackendState {
    explicit FakeBackendState(int value) : value(value) {}
    int value = 0;
};

class FakeCompiledKernel final : public CompiledKernelBase {
public:
    explicit FakeCompiledKernel(uint32_t arg_count = 0) : CompiledKernelBase(arg_count) {}

    explicit FakeCompiledKernel(std::shared_ptr<const KernelBindingPlan> binding_plan)
        : CompiledKernelBase(std::move(binding_plan)) {}

    FakeCompiledKernel(std::shared_ptr<const KernelBindingPlan> binding_plan, std::shared_ptr<void> prepared_binding_cache)
        : CompiledKernelBase(std::move(binding_plan), std::move(prepared_binding_cache)) {}

    size_t clamp_threadgroup_size(size_t desired) const override {
        return desired;
    }

    std::shared_ptr<ICompiledKernel> fork() const override {
        return std::make_shared<FakeCompiledKernel>(binding_plan(), prepared_binding_cache());
    }

    void execute(GpuCommandBufferHandle,
                 const KernelDispatch&,
                 const std::vector<KernelArg>&,
                 const KernelExecutionHooks*) override {}

    uint32_t resolve(const std::vector<KernelArg>& args, const char* label) const {
        return resolve_runtime_arg_count(args, label);
    }

    KernelBindingTable bindings(const std::vector<KernelArg>& args, const char* label) const {
        return materialize_runtime_bindings(args, label);
    }

    std::shared_ptr<const PreparedKernelBindings> prepared(const std::vector<KernelArg>& args, const char* label) const {
        return get_or_create_prepared_bindings(args, label);
    }

    mutable size_t prepared_binding_create_count = 0;

private:
    std::shared_ptr<const PreparedKernelBindings> create_prepared_bindings(
        const KernelBindingTable& bindings) const override {
        ++prepared_binding_create_count;
        return CompiledKernelBase::create_prepared_bindings(bindings);
    }
};

compiler::GfxCompileResult compile_metal_without_graph_pipeline(
    const std::shared_ptr<const ov::Model>& model) {
    const auto& registry = compiler::BackendRegistry::default_registry();
    const auto metal = registry.resolve(GpuBackend::Metal);
    OPENVINO_ASSERT(metal);

    compiler::GfxCompileResult compile_result;
    compile_result.target = metal->target();
    compile_result.transformed_model = model;
    compile_result.lowering_plan =
        metal->lowering_planner().plan(model, metal->legalizer());
    compile_result.manifest =
        compiler::ManifestBuilder{}.build(compile_result.lowering_plan);
    compile_result.executable = compiler::ExecutableBundleBuilder(
        [metal](compiler::KernelArtifactDescriptor& descriptor,
                const compiler::PlannedOperation& op) {
            return metal->materialize_artifact_payload(descriptor, op);
        }).build(compile_result.manifest, compile_result.lowering_plan);
    compile_result.unsupported = compile_result.lowering_plan.unsupported;
    return compile_result;
}

void expect_metal_compiler_owned_msl_payload(
    const compiler::GfxCompileResult& compile_result,
    const char* op_family,
    const char* kernel_id,
    const char* entry_point,
    uint32_t expected_arg_count,
    uint32_t expected_output_arg_count,
    const char* source_needle) {
    ASSERT_TRUE(compile_result.supported());
    ASSERT_TRUE(compile_result.executable.verify().valid());

    const auto artifact_it = std::find_if(
        compile_result.executable.artifact_descriptors.begin(),
        compile_result.executable.artifact_descriptors.end(),
        [op_family](const compiler::KernelArtifactDescriptor& artifact) {
            return artifact.kernel.op_family == op_family;
        });
    ASSERT_NE(artifact_it, compile_result.executable.artifact_descriptors.end());
    EXPECT_EQ(artifact_it->kernel.origin,
              compiler::KernelArtifactOrigin::Generated);
    EXPECT_EQ(artifact_it->payload_kind,
              compiler::KernelArtifactPayloadKind::MslSource);
    EXPECT_EQ(artifact_it->kernel.kernel_id, kernel_id);
    EXPECT_EQ(artifact_it->entry_point, entry_point);
    EXPECT_EQ(artifact_it->abi_arg_count, expected_arg_count);
    EXPECT_EQ(artifact_it->abi_output_arg_count, expected_output_arg_count);

    const auto payload =
        compile_result.executable.find_artifact_payload(artifact_it->artifact_key);
    ASSERT_TRUE(payload);
    EXPECT_EQ(payload->payload_kind(),
              compiler::KernelArtifactPayloadKind::MslSource);
    EXPECT_EQ(payload->backend_domain(), "metal");
    EXPECT_EQ(payload->source_id(), artifact_it->kernel.kernel_id);
    EXPECT_EQ(payload->entry_point(), artifact_it->entry_point);
    const auto* source_payload =
        dynamic_cast<const GfxKernelSourcePayload*>(payload.get());
    ASSERT_NE(source_payload, nullptr);
    EXPECT_NE(std::string(source_payload->source().source).find(source_needle),
              std::string::npos);

    const auto runtime_descriptor =
        RuntimeExecutableDescriptorBuilder{}.build(compile_result.executable);
    ASSERT_TRUE(runtime_descriptor.verify(compile_result.executable).valid());
    const auto stage_it = std::find_if(
        runtime_descriptor.stages.begin(),
        runtime_descriptor.stages.end(),
        [op_family](const RuntimeStageExecutableDescriptor& stage) {
            return stage.op_family == op_family;
        });
    ASSERT_NE(stage_it, runtime_descriptor.stages.end());
    EXPECT_EQ(stage_it->payload, payload);
    EXPECT_EQ(stage_it->payload_kind,
              compiler::KernelArtifactPayloadKind::MslSource);
    EXPECT_EQ(stage_it->abi_arg_count, artifact_it->abi_arg_count);
    EXPECT_EQ(stage_it->abi_output_arg_count,
              artifact_it->abi_output_arg_count);
}

void expect_metal_compiler_owned_vendor_payload(
    const compiler::GfxCompileResult& compile_result,
    const char* op_family,
    const char* kernel_id,
    const char* entry_point,
    GfxAppleMpsVendorPrimitiveKind expected_kind,
    uint32_t expected_arg_count,
    uint32_t expected_output_arg_count,
    size_t expected_input_desc_count,
    size_t expected_output_desc_count) {
    ASSERT_TRUE(compile_result.supported());
    ASSERT_TRUE(compile_result.executable.verify().valid());

    const auto artifact_it = std::find_if(
        compile_result.executable.artifact_descriptors.begin(),
        compile_result.executable.artifact_descriptors.end(),
        [op_family](const compiler::KernelArtifactDescriptor& artifact) {
            return artifact.kernel.op_family == op_family;
        });
    ASSERT_NE(artifact_it, compile_result.executable.artifact_descriptors.end());
    EXPECT_EQ(artifact_it->kernel.origin,
              compiler::KernelArtifactOrigin::VendorPrimitive);
    EXPECT_EQ(artifact_it->payload_kind,
              compiler::KernelArtifactPayloadKind::VendorDescriptor);
    EXPECT_EQ(artifact_it->kernel.kernel_id, kernel_id);
    EXPECT_EQ(artifact_it->entry_point, entry_point);
    EXPECT_EQ(artifact_it->abi_arg_count, expected_arg_count);
    EXPECT_EQ(artifact_it->abi_output_arg_count, expected_output_arg_count);

    const auto payload =
        compile_result.executable.find_artifact_payload(artifact_it->artifact_key);
    ASSERT_TRUE(payload);
    EXPECT_EQ(payload->payload_kind(),
              compiler::KernelArtifactPayloadKind::VendorDescriptor);
    EXPECT_EQ(payload->backend_domain(), "metal");
    EXPECT_EQ(payload->source_id(), artifact_it->kernel.kernel_id);
    EXPECT_EQ(payload->entry_point(), artifact_it->entry_point);
    const auto* vendor_payload =
        dynamic_cast<const compiler::GfxMetalVendorPrimitiveArtifactPayload*>(
            payload.get());
    ASSERT_NE(vendor_payload, nullptr);
    const auto& contract = vendor_payload->contract();
    ASSERT_TRUE(contract.valid);
    EXPECT_EQ(contract.descriptor.kind, expected_kind);
    EXPECT_TRUE(contract.external_buffer_abi.valid);
    EXPECT_EQ(contract.external_buffer_abi.buffer_count,
              expected_arg_count);
    EXPECT_EQ(contract.external_buffer_abi.output_buffer_count,
              expected_output_arg_count);
    EXPECT_EQ(contract.input_descs.size(), expected_input_desc_count);
    EXPECT_EQ(contract.output_descs.size(), expected_output_desc_count);

    const auto runtime_descriptor =
        RuntimeExecutableDescriptorBuilder{}.build(compile_result.executable);
    ASSERT_TRUE(runtime_descriptor.verify(compile_result.executable).valid());
    const auto stage_it = std::find_if(
        runtime_descriptor.stages.begin(),
        runtime_descriptor.stages.end(),
        [op_family](const RuntimeStageExecutableDescriptor& stage) {
            return stage.op_family == op_family;
        });
    ASSERT_NE(stage_it, runtime_descriptor.stages.end());
    EXPECT_EQ(stage_it->payload, payload);
    EXPECT_EQ(stage_it->payload_kind,
              compiler::KernelArtifactPayloadKind::VendorDescriptor);
    EXPECT_EQ(stage_it->origin,
              compiler::KernelArtifactOrigin::VendorPrimitive);
    EXPECT_EQ(stage_it->abi_arg_count, artifact_it->abi_arg_count);
    EXPECT_EQ(stage_it->abi_output_arg_count,
              artifact_it->abi_output_arg_count);
}

TEST(GpuBackendBaseTest, BackendTargetIsImmutableAndCapabilityDriven) {
    const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
    EXPECT_EQ(target.backend(), GpuBackend::Metal);
    EXPECT_NE(target.fingerprint().find("backend=metal"), std::string::npos);
    EXPECT_TRUE(target.is_compatible_with_fingerprint(target.fingerprint()));

    compiler::BackendCapabilities capabilities(target, compiler::make_metal_operation_support_policy());
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3});
    const auto support = capabilities.query_operation({parameter});
    EXPECT_TRUE(support.semantic_legal);
    EXPECT_EQ(support.preferred_route_kind, compiler::LoweringRouteKind::Common);
}

TEST(GpuBackendBaseTest, BackendRegistryResolvesCompiledBackendModules) {
    const auto& registry = compiler::BackendRegistry::default_registry();
    const auto targets = registry.available_targets();
#if GFX_BACKEND_METAL_AVAILABLE || GFX_BACKEND_OPENCL_AVAILABLE
    EXPECT_FALSE(targets.empty());
#endif

#if GFX_BACKEND_METAL_AVAILABLE
    const auto metal = registry.resolve(GpuBackend::Metal);
    ASSERT_TRUE(metal);
    EXPECT_EQ(metal->id(), "metal");
    EXPECT_EQ(metal->target().backend(), GpuBackend::Metal);
    EXPECT_TRUE(metal->kernel_registry().audit().valid());
    EXPECT_GE(metal->kernel_registry().route_count(compiler::LoweringRouteKind::Common), 1u);

    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3});
    EXPECT_TRUE(metal->capabilities().supports_node(parameter));
#else
    EXPECT_FALSE(registry.resolve(GpuBackend::Metal));
#endif
}

TEST(GpuBackendBaseTest, BackendModuleOwnsLegalizerAndLoweringPlanner) {
    const auto& registry = compiler::BackendRegistry::default_registry();
#if GFX_BACKEND_METAL_AVAILABLE
    const auto metal = registry.resolve(GpuBackend::Metal);
    ASSERT_TRUE(metal);

    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3});
    auto result = std::make_shared<ov::op::v0::Result>(parameter);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{parameter});

    const auto plan = metal->lowering_planner().plan(model, metal->legalizer());
    EXPECT_TRUE(plan.executable());
    EXPECT_EQ(plan.target.backend(), GpuBackend::Metal);
    EXPECT_GE(plan.route_count(compiler::LoweringRouteKind::Common), 2u);
    ASSERT_FALSE(plan.operations.empty());
    EXPECT_TRUE(plan.operations.front().kernel_unit.valid());
    EXPECT_EQ(plan.operations.front().kernel_unit.route_kind(),
              compiler::LoweringRouteKind::Common);
    EXPECT_EQ(plan.operations.front().kernel_unit.kind(),
              compiler::KernelUnitKind::Common);

    const auto manifest = compiler::ManifestBuilder{}.build(plan);
    EXPECT_TRUE(manifest.valid());
    EXPECT_TRUE(manifest.verify().valid());
    EXPECT_EQ(manifest.schema_version, 2u);
    EXPECT_EQ(manifest.target_fingerprint, metal->target().fingerprint());
    ASSERT_EQ(manifest.stages.size(), plan.operations.size());
    EXPECT_EQ(manifest.route_count(compiler::LoweringRouteKind::Common),
              plan.route_count(compiler::LoweringRouteKind::Common));
    const auto& first_stage = manifest.stages.front();
    ASSERT_FALSE(first_stage.outputs.empty());
    EXPECT_FALSE(first_stage.memory.hidden_host_copy_allowed);
    EXPECT_EQ(first_stage.dispatch.backend_domain, metal->target().backend_id());
    EXPECT_EQ(first_stage.dispatch.kernel_unit_id,
              plan.operations.front().kernel_unit.id());
    EXPECT_EQ(first_stage.dispatch.kernel_unit_id, first_stage.kernel_unit_id);
    EXPECT_EQ(first_stage.dispatch.kernel_unit_kind, first_stage.kernel_unit_kind);
    EXPECT_EQ(first_stage.kernel_unit_kind, "common");
    EXPECT_FALSE(first_stage.outputs.front().lifetime_class.empty());

    auto broken_manifest = manifest;
    broken_manifest.stages.front().memory.hidden_host_copy_allowed = true;
    EXPECT_FALSE(broken_manifest.verify().valid());
#else
    EXPECT_FALSE(registry.resolve(GpuBackend::Metal));
#endif
}

TEST(GpuBackendBaseTest, ManifestBuilderEmitsRuntimeShapeContractForDynamicTensors) {
    const auto& registry = compiler::BackendRegistry::default_registry();
#if GFX_BACKEND_METAL_AVAILABLE
    const auto metal = registry.resolve(GpuBackend::Metal);
    ASSERT_TRUE(metal);

    auto parameter = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::PartialShape{ov::Dimension::dynamic(), 3});
    auto result = std::make_shared<ov::op::v0::Result>(parameter);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{parameter});

    const auto plan = metal->lowering_planner().plan(model, metal->legalizer());
    ASSERT_TRUE(plan.executable());

    const auto manifest = compiler::ManifestBuilder{}.build(plan);
    ASSERT_TRUE(manifest.verify().valid());
    ASSERT_FALSE(manifest.stages.empty());

    bool found_shape_param = false;
    for (const auto& stage : manifest.stages) {
        EXPECT_EQ(stage.runtime_params.params.size(),
                  stage.runtime_params.runtime_param_names.size());
        for (const auto& param : stage.runtime_params.params) {
            if (param.kind == compiler::RuntimeParamKind::Shape) {
                found_shape_param = true;
                EXPECT_EQ(param.abi_type, "shape_i64");
                EXPECT_FALSE(param.source_tensor.empty());
            }
        }
    }
    EXPECT_TRUE(found_shape_param);
#else
    EXPECT_FALSE(registry.resolve(GpuBackend::Metal));
#endif
}

TEST(GpuBackendBaseTest, CompilerServiceBuildsManifestBundle) {
#if GFX_BACKEND_METAL_AVAILABLE
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3});
    auto result = std::make_shared<ov::op::v0::Result>(parameter);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{parameter});

    const compiler::GfxCompilerService compiler_service;
    const auto compile_result = compiler_service.compile(
        {model, compiler::BackendTarget::from_backend(GpuBackend::Metal)});
    EXPECT_TRUE(compile_result.supported());
    EXPECT_TRUE(compile_result.manifest.valid());
    EXPECT_TRUE(compile_result.manifest.verify().valid());
    EXPECT_TRUE(compile_result.executable.valid());
    EXPECT_TRUE(compile_result.executable.verify().valid());
    EXPECT_EQ(compile_result.manifest.target_fingerprint, compile_result.target.fingerprint());
    EXPECT_EQ(compile_result.executable.target_fingerprint, compile_result.target.fingerprint());
    EXPECT_EQ(compile_result.manifest.stages.size(), compile_result.lowering_plan.operations.size());
    EXPECT_EQ(compile_result.executable.stages.size(), compile_result.manifest.stages.size());
    EXPECT_EQ(compile_result.executable.artifact_descriptors.size(),
              compile_result.manifest.stages.size());
    ASSERT_FALSE(compile_result.executable.stages.empty());
    ASSERT_FALSE(compile_result.executable.artifact_descriptors.empty());
    EXPECT_EQ(compile_result.executable.stages.front().kernel_unit_id,
              compile_result.manifest.stages.front().kernel_unit_id);
    const auto& artifact = compile_result.executable.artifact_descriptors.front();
    EXPECT_EQ(artifact.stage_record_key,
              compile_result.manifest.stages.front().stable_record_key);
    EXPECT_EQ(artifact.kernel.kernel_id,
              compile_result.manifest.stages.front().kernel_unit_id);
    EXPECT_EQ(artifact.kernel.backend_domain,
              compile_result.manifest.stages.front().backend_domain);
    EXPECT_FALSE(artifact.manifest_ref.empty());
    EXPECT_FALSE(artifact.abi_fingerprint.empty());
    EXPECT_FALSE(artifact.artifact_key.empty());
    EXPECT_EQ(artifact.kernel.origin, compiler::KernelArtifactOrigin::Common);
    EXPECT_EQ(artifact.payload_kind, compiler::KernelArtifactPayloadKind::None);
    const auto runtime_descriptor =
        RuntimeExecutableDescriptorBuilder{}.build(compile_result.executable);
    EXPECT_TRUE(runtime_descriptor.valid(compile_result.executable));
    EXPECT_EQ(runtime_descriptor.target_fingerprint,
              compile_result.executable.target_fingerprint);
    ASSERT_EQ(runtime_descriptor.stages.size(),
              compile_result.executable.stages.size());
    EXPECT_EQ(runtime_descriptor.stages.front().stage_record_key,
              compile_result.executable.stages.front().stage_record_key);
    EXPECT_EQ(runtime_descriptor.stages.front().manifest_ref,
              artifact.manifest_ref);
    EXPECT_EQ(runtime_descriptor.stages.front().abi_fingerprint,
              artifact.abi_fingerprint);
    EXPECT_EQ(runtime_descriptor.stages.front().artifact_key,
              artifact.artifact_key);
    EXPECT_EQ(runtime_descriptor.stages.front().kernel_id,
              artifact.kernel.kernel_id);
    auto broken_executable = compile_result.executable;
    broken_executable.stages.front().kernel_unit_id.clear();
    EXPECT_FALSE(broken_executable.verify().valid());
    auto broken_artifact = compile_result.executable;
    broken_artifact.artifact_descriptors.front().kernel.kernel_id.clear();
    EXPECT_FALSE(broken_artifact.verify().valid());
    auto broken_runtime_descriptor = runtime_descriptor;
    broken_runtime_descriptor.stages.front().artifact_key.clear();
    EXPECT_FALSE(broken_runtime_descriptor.verify(compile_result.executable).valid());
#endif
}

TEST(GpuBackendBaseTest, RuntimeExecutableDescriptorKeepsGeneratedOpenClRouteImmutable) {
    compiler::ManifestBundle manifest;
    manifest.target_fingerprint =
        compiler::BackendTarget::from_backend(GpuBackend::OpenCL).fingerprint();

    compiler::StageRecord stage;
    stage.stage_id = 0;
    stage.stable_record_key = 42;
    stage.source_node_name = "conv";
    stage.normalized_op_family = "Convolution";
    stage.execution_kind = compiler::LoweringRouteKind::GeneratedKernel;
    stage.backend_domain = "opencl";
    stage.kernel_unit_id = "opencl_generated_kernel";
    stage.kernel_unit_kind =
        std::string(compiler::kernel_unit_kind_to_string(compiler::KernelUnitKind::GeneratedKernel));

    compiler::TensorContract input;
    input.logical_name = "conv.input0";
    input.role = compiler::TensorContractRole::TensorInput;
    input.element_type = "f32";
    input.partial_shape = "[1,3,5,5]";
    input.lifetime_class = "producer_or_external";
    stage.inputs.push_back(input);

    compiler::TensorContract output;
    output.logical_name = "conv.output0";
    output.role = compiler::TensorContractRole::TensorOutput;
    output.element_type = "f32";
    output.partial_shape = "[1,8,3,3]";
    output.lifetime_class = "stage_output";
    stage.outputs.push_back(output);

    stage.dispatch.execution_kind = stage.execution_kind;
    stage.dispatch.backend_domain = stage.backend_domain;
    stage.dispatch.kernel_unit_id = stage.kernel_unit_id;
    stage.dispatch.kernel_unit_kind = stage.kernel_unit_kind;
    stage.dispatch.dispatch_source = "unit_test_generated_kernel";
    stage.memory.alias_group = "stage_0";
    manifest.stages.push_back(stage);

    ASSERT_TRUE(manifest.verify().valid());

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    ASSERT_TRUE(executable.verify().valid());
    ASSERT_EQ(executable.artifact_descriptors.size(), 1u);
    const auto& artifact = executable.artifact_descriptors.front();
    EXPECT_EQ(artifact.kernel.origin, compiler::KernelArtifactOrigin::Generated);
    EXPECT_EQ(artifact.payload_kind, compiler::KernelArtifactPayloadKind::OpenClSource);
    EXPECT_EQ(artifact.kernel.backend_domain, "opencl");
    EXPECT_EQ(artifact.entry_point, "opencl_generated_kernel");
    EXPECT_FALSE(artifact.manifest_ref.empty());
    EXPECT_FALSE(artifact.abi_fingerprint.empty());
    EXPECT_FALSE(artifact.artifact_key.empty());

    const auto runtime_descriptor = RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(runtime_descriptor.verify(executable).valid());
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().origin,
              compiler::KernelArtifactOrigin::Generated);
    EXPECT_EQ(runtime_descriptor.stages.front().payload_kind,
              compiler::KernelArtifactPayloadKind::OpenClSource);
    EXPECT_EQ(runtime_descriptor.stages.front().backend_domain, "opencl");

    auto broken_runtime_descriptor = runtime_descriptor;
    broken_runtime_descriptor.stages.front().origin =
        compiler::KernelArtifactOrigin::BackendLowering;
    EXPECT_FALSE(broken_runtime_descriptor.verify(executable).valid());
}

TEST(GpuBackendBaseTest, OpenClCompilerBundleOwnsSourceArtifactPayload) {
    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{4});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{4});
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{lhs, rhs});

    const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
    const compiler::BackendCapabilities capabilities(
        target,
        compiler::make_opencl_operation_support_policy());
    const compiler::OperationLegalizer legalizer(capabilities);
    const compiler::LoweringPlanner planner(target,
                                            compiler::make_opencl_kernel_registry(target));
    const auto plan = planner.plan(model, legalizer);
    ASSERT_TRUE(plan.executable());
    EXPECT_EQ(plan.route_count(compiler::LoweringRouteKind::HandwrittenKernelException),
              1u);

    const auto manifest = compiler::ManifestBuilder{}.build(plan);
    ASSERT_TRUE(manifest.verify().valid());
    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest,
                                                                      plan);
    ASSERT_TRUE(executable.verify().valid());

    const auto artifact_it = std::find_if(
        executable.artifact_descriptors.begin(),
        executable.artifact_descriptors.end(),
        [](const compiler::KernelArtifactDescriptor& artifact) {
            return artifact.kernel.op_family == "Add";
        });
    ASSERT_NE(artifact_it, executable.artifact_descriptors.end());
    EXPECT_EQ(artifact_it->kernel.origin,
              compiler::KernelArtifactOrigin::HandwrittenException);
    EXPECT_EQ(artifact_it->payload_kind,
              compiler::KernelArtifactPayloadKind::OpenClSource);
    EXPECT_EQ(artifact_it->kernel.kernel_id,
              "opencl/baseline/eltwise_binary_f32");
    EXPECT_EQ(artifact_it->entry_point,
              "gfx_opencl_baseline_binary_f32");

    const auto payload = executable.find_artifact_payload(artifact_it->artifact_key);
    ASSERT_TRUE(payload);
    EXPECT_EQ(payload->payload_kind(),
              compiler::KernelArtifactPayloadKind::OpenClSource);
    EXPECT_EQ(payload->backend_domain(), "opencl");
    EXPECT_EQ(payload->source_id(), artifact_it->kernel.kernel_id);
    EXPECT_EQ(payload->entry_point(), artifact_it->entry_point);
    const auto* source_payload =
        dynamic_cast<const GfxOpenClSourceArtifactPayload*>(payload.get());
    ASSERT_NE(source_payload, nullptr);
    EXPECT_TRUE(source_payload->artifact().valid);

    const auto runtime_descriptor =
        RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(runtime_descriptor.verify(executable).valid());
    const auto stage_it = std::find_if(
        runtime_descriptor.stages.begin(),
        runtime_descriptor.stages.end(),
        [](const RuntimeStageExecutableDescriptor& stage) {
            return stage.op_family == "Add";
        });
    ASSERT_NE(stage_it, runtime_descriptor.stages.end());
    EXPECT_EQ(stage_it->payload, payload);
    EXPECT_EQ(stage_it->entry_point, artifact_it->entry_point);
}

TEST(GpuBackendBaseTest, OpenClCompilerBundleOwnsSoftmaxKernelFilePayload) {
    auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{2, 3, 4});
    auto softmax = std::make_shared<ov::op::v8::Softmax>(input, -1);
    auto result = std::make_shared<ov::op::v0::Result>(softmax);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{input});

    const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
    const compiler::BackendCapabilities capabilities(
        target,
        compiler::make_opencl_operation_support_policy());
    const compiler::OperationLegalizer legalizer(capabilities);
    const compiler::LoweringPlanner planner(target,
                                            compiler::make_opencl_kernel_registry(target));
    const auto plan = planner.plan(model, legalizer);
    ASSERT_TRUE(plan.executable());
    EXPECT_EQ(plan.route_count(compiler::LoweringRouteKind::HandwrittenKernelException),
              1u);

    const auto manifest = compiler::ManifestBuilder{}.build(plan);
    ASSERT_TRUE(manifest.verify().valid());
    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest,
                                                                      plan);
    ASSERT_TRUE(executable.verify().valid());

    const auto artifact_it = std::find_if(
        executable.artifact_descriptors.begin(),
        executable.artifact_descriptors.end(),
        [](const compiler::KernelArtifactDescriptor& artifact) {
            return artifact.kernel.op_family == "Softmax";
        });
    ASSERT_NE(artifact_it, executable.artifact_descriptors.end());
    EXPECT_EQ(artifact_it->kernel.origin,
              compiler::KernelArtifactOrigin::HandwrittenException);
    EXPECT_EQ(artifact_it->payload_kind,
              compiler::KernelArtifactPayloadKind::OpenClSource);
    EXPECT_EQ(artifact_it->kernel.kernel_id,
              "opencl/baseline/softmax_f32");
    EXPECT_EQ(artifact_it->entry_point,
              "gfx_opencl_baseline_softmax_f32");

    const auto payload = executable.find_artifact_payload(artifact_it->artifact_key);
    ASSERT_TRUE(payload);
    EXPECT_EQ(payload->payload_kind(),
              compiler::KernelArtifactPayloadKind::OpenClSource);
    EXPECT_EQ(payload->backend_domain(), "opencl");
    EXPECT_EQ(payload->source_id(), artifact_it->kernel.kernel_id);
    EXPECT_EQ(payload->entry_point(), artifact_it->entry_point);
    const auto* source_payload =
        dynamic_cast<const GfxOpenClSourceArtifactPayload*>(payload.get());
    ASSERT_NE(source_payload, nullptr);
    ASSERT_TRUE(source_payload->artifact().valid);
    EXPECT_EQ(source_payload->artifact().source,
              opencl_baseline_softmax_f32_kernel_source().source);
    EXPECT_NE(source_payload->artifact().source.find(
              "gfx_opencl_baseline_softmax_dynamic_f32"),
              std::string::npos);
}

TEST(GpuBackendBaseTest, OpenClCompilerBundleOwnsInterpolateGeneratedKernelFilePayload) {
    auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{1, 4, 16, 16});
    auto output_shape = ov::op::v0::Constant::create(
        ov::element::i64,
        ov::Shape{2},
        std::vector<int64_t>{32, 32});
    ov::op::v0::Interpolate::Attributes attrs;
    attrs.axes = ov::AxisSet{2, 3};
    attrs.mode = "linear";
    attrs.align_corners = false;
    auto interpolate =
        std::make_shared<ov::op::v0::Interpolate>(input, output_shape, attrs);
    auto result = std::make_shared<ov::op::v0::Result>(interpolate);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{input});

    const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
    const compiler::BackendCapabilities capabilities(
        target,
        compiler::make_opencl_operation_support_policy());
    const compiler::OperationLegalizer legalizer(capabilities);
    const compiler::LoweringPlanner planner(target,
                                            compiler::make_opencl_kernel_registry(target));
    const auto plan = planner.plan(model, legalizer);
    ASSERT_TRUE(plan.executable());
    EXPECT_EQ(plan.route_count(compiler::LoweringRouteKind::GeneratedKernel),
              1u);
    EXPECT_EQ(plan.route_count(compiler::LoweringRouteKind::HandwrittenKernelException),
              0u);

    const auto manifest = compiler::ManifestBuilder{}.build(plan);
    ASSERT_TRUE(manifest.verify().valid());
    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest,
                                                                      plan);
    ASSERT_TRUE(executable.verify().valid());

    const auto artifact_it = std::find_if(
        executable.artifact_descriptors.begin(),
        executable.artifact_descriptors.end(),
        [](const compiler::KernelArtifactDescriptor& artifact) {
            return artifact.kernel.op_family == "Interpolate";
        });
    ASSERT_NE(artifact_it, executable.artifact_descriptors.end());
    EXPECT_EQ(artifact_it->kernel.origin,
              compiler::KernelArtifactOrigin::Generated);
    EXPECT_EQ(artifact_it->payload_kind,
              compiler::KernelArtifactPayloadKind::OpenClSource);
    EXPECT_EQ(artifact_it->kernel.kernel_id,
              "opencl/generated/interpolate_f32");
    EXPECT_EQ(artifact_it->entry_point,
              "gfx_opencl_generated_interpolate_f32");
    EXPECT_EQ(artifact_it->abi_arg_count, 13u);
    EXPECT_EQ(artifact_it->abi_output_arg_count, 1u);

    const auto payload =
        executable.find_artifact_payload(artifact_it->artifact_key);
    ASSERT_TRUE(payload);
    EXPECT_EQ(payload->payload_kind(),
              compiler::KernelArtifactPayloadKind::OpenClSource);
    EXPECT_EQ(payload->backend_domain(), "opencl");
    EXPECT_EQ(payload->source_id(), artifact_it->kernel.kernel_id);
    EXPECT_EQ(payload->entry_point(), artifact_it->entry_point);
    const auto* source_payload =
        dynamic_cast<const GfxOpenClSourceArtifactPayload*>(payload.get());
    ASSERT_NE(source_payload, nullptr);
    ASSERT_TRUE(source_payload->artifact().valid);
    EXPECT_EQ(source_payload->artifact().source,
              opencl_generated_interpolate_f32_kernel_source().source);
    EXPECT_NE(source_payload->artifact().source.find(
                  "gfx_opencl_generated_interpolate_f32"),
              std::string::npos);
}

TEST(GpuBackendBaseTest, OpenClCompilerPlansPool2DAsGeneratedKernel) {
    auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{1, 4, 16, 16});
    auto pool = std::make_shared<ov::op::v1::MaxPool>(
        input,
        ov::Strides{2, 2},
        ov::Shape{0, 0},
        ov::Shape{0, 0},
        ov::Shape{2, 2},
        ov::op::RoundingType::FLOOR);
    auto result = std::make_shared<ov::op::v0::Result>(pool);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{input});

    const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
    const compiler::BackendCapabilities capabilities(
        target,
        compiler::make_opencl_operation_support_policy());
    const compiler::OperationLegalizer legalizer(capabilities);
    const compiler::LoweringPlanner planner(target,
                                            compiler::make_opencl_kernel_registry(target));
    const auto plan = planner.plan(model, legalizer);
    ASSERT_TRUE(plan.executable());
    EXPECT_EQ(plan.route_count(compiler::LoweringRouteKind::GeneratedKernel),
              1u);
    EXPECT_EQ(plan.route_count(compiler::LoweringRouteKind::HandwrittenKernelException),
              0u);

    const auto manifest = compiler::ManifestBuilder{}.build(plan);
    ASSERT_TRUE(manifest.verify().valid());
    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest,
                                                                      plan);
    ASSERT_TRUE(executable.verify().valid());

    const auto artifact_it = std::find_if(
        executable.artifact_descriptors.begin(),
        executable.artifact_descriptors.end(),
        [](const compiler::KernelArtifactDescriptor& artifact) {
            return artifact.kernel.op_family == "MaxPool";
        });
    ASSERT_NE(artifact_it, executable.artifact_descriptors.end());
    EXPECT_EQ(artifact_it->kernel.origin,
              compiler::KernelArtifactOrigin::Generated);
    EXPECT_EQ(artifact_it->payload_kind,
              compiler::KernelArtifactPayloadKind::OpenClSource);
    EXPECT_EQ(artifact_it->kernel.kernel_id, "opencl_generated_kernel");
    EXPECT_EQ(artifact_it->entry_point, "opencl_generated_kernel");
    EXPECT_FALSE(executable.find_artifact_payload(artifact_it->artifact_key));
}

TEST(GpuBackendBaseTest, MetalCompilerBundleOwnsShapeOfMslPayload) {
#if GFX_BACKEND_METAL_AVAILABLE
    auto parameter = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::PartialShape{ov::Dimension::dynamic(), 3});
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(parameter,
                                                          ov::element::i64);
    auto result = std::make_shared<ov::op::v0::Result>(shape_of);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{parameter});

    const compiler::GfxCompilerService compiler_service;
    const auto compile_result = compiler_service.compile(
        {model, compiler::BackendTarget::from_backend(GpuBackend::Metal)});
    expect_metal_compiler_owned_msl_payload(compile_result,
                                            "ShapeOf",
                                            "metal/generated/shapeof",
                                            "shapeof_kernel",
                                            4u,
                                            1u,
                                            "kernel void shapeof_kernel");
#endif
}

TEST(GpuBackendBaseTest, MetalCompilerBundleOwnsRangeMslPayload) {
#if GFX_BACKEND_METAL_AVAILABLE
    auto start = ov::op::v0::Constant::create(ov::element::f32,
                                              ov::Shape{},
                                              {0.0f});
    auto stop = ov::op::v0::Constant::create(ov::element::f32,
                                             ov::Shape{},
                                             {10.0f});
    auto step = ov::op::v0::Constant::create(ov::element::f32,
                                             ov::Shape{},
                                             {2.0f});
    auto range =
        std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::f32);
    auto result = std::make_shared<ov::op::v0::Result>(range);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{});

    const auto compile_result = compile_metal_without_graph_pipeline(model);
    expect_metal_compiler_owned_msl_payload(compile_result,
                                            "Range",
                                            "metal/generated/range",
                                            "range_kernel",
                                            5u,
                                            1u,
                                            "kernel void range_kernel");
#endif
}

TEST(GpuBackendBaseTest, MetalCompilerBundleOwnsTileMslPayload) {
#if GFX_BACKEND_METAL_AVAILABLE
    auto parameter = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{2, 3});
    auto repeats = ov::op::v0::Constant::create(ov::element::i64,
                                                ov::Shape{2},
                                                {1, 2});
    auto tile = std::make_shared<ov::op::v0::Tile>(parameter, repeats);
    auto result = std::make_shared<ov::op::v0::Result>(tile);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{parameter});

    const compiler::GfxCompilerService compiler_service;
    const auto compile_result = compiler_service.compile(
        {model, compiler::BackendTarget::from_backend(GpuBackend::Metal)});
    expect_metal_compiler_owned_msl_payload(compile_result,
                                            "Tile",
                                            "metal/generated/tile",
                                            "tile_kernel",
                                            8u,
                                            1u,
                                            "kernel void tile_kernel");
#endif
}

TEST(GpuBackendBaseTest, MetalCompilerBundleOwnsConcatMslPayload) {
#if GFX_BACKEND_METAL_AVAILABLE
    auto lhs = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{1, 2, 4});
    auto rhs = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{1, 3, 4});
    auto concat = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{lhs, rhs},
        1);
    auto result = std::make_shared<ov::op::v0::Result>(concat);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{lhs, rhs});

    const auto compile_result = compile_metal_without_graph_pipeline(model);
    expect_metal_compiler_owned_msl_payload(compile_result,
                                            "Concat",
                                            "metal/generated/concat",
                                            "concat_kernel",
                                            3u,
                                            1u,
                                            "kernel void concat_kernel");
#endif
}

TEST(GpuBackendBaseTest, MetalCompilerBundleOwnsSplitMslPayload) {
#if GFX_BACKEND_METAL_AVAILABLE
    auto parameter = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{1, 6, 4});
    auto axis = ov::op::v0::Constant::create(ov::element::i64,
                                             ov::Shape{},
                                             {1});
    auto split = std::make_shared<ov::op::v1::Split>(parameter, axis, 3);
    ov::ResultVector results;
    for (size_t i = 0; i < split->get_output_size(); ++i) {
        results.push_back(std::make_shared<ov::op::v0::Result>(
            split->output(i)));
    }
    auto model = std::make_shared<ov::Model>(results,
                                             ov::ParameterVector{parameter});

    const auto compile_result = compile_metal_without_graph_pipeline(model);
    expect_metal_compiler_owned_msl_payload(compile_result,
                                            "Split",
                                            "metal/generated/split",
                                            "split_kernel",
                                            4u,
                                            3u,
                                            "kernel void split_kernel");
#endif
}

TEST(GpuBackendBaseTest, MetalCompilerBundleOwnsSliceMslPayload) {
#if GFX_BACKEND_METAL_AVAILABLE
    auto data = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{2, 3, 4});
    auto starts = ov::op::v0::Constant::create(ov::element::i64,
                                               ov::Shape{3},
                                               {0, 1, 0});
    auto stops = ov::op::v0::Constant::create(ov::element::i64,
                                              ov::Shape{3},
                                              {2, 3, 4});
    auto steps = ov::op::v0::Constant::create(ov::element::i64,
                                              ov::Shape{3},
                                              {1, 1, 2});
    auto axes = ov::op::v0::Constant::create(ov::element::i64,
                                             ov::Shape{3},
                                             {0, 1, 2});
    auto slice = std::make_shared<ov::op::v8::Slice>(data,
                                                     starts,
                                                     stops,
                                                     steps,
                                                     axes);
    auto result = std::make_shared<ov::op::v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{data});

    const auto compile_result = compile_metal_without_graph_pipeline(model);
    expect_metal_compiler_owned_msl_payload(compile_result,
                                            "Slice",
                                            "metal/generated/slice",
                                            "slice_kernel",
                                            2u,
                                            1u,
                                            "kernel void slice_kernel");
#endif
}

TEST(GpuBackendBaseTest, MetalCompilerBundleOwnsMpsSoftmaxVendorPayload) {
#if GFX_BACKEND_METAL_AVAILABLE
    auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{1, 8});
    auto softmax = std::make_shared<ov::op::v8::Softmax>(input, 1);
    auto result = std::make_shared<ov::op::v0::Result>(softmax);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{input});

    const auto compile_result = compile_metal_without_graph_pipeline(model);
    expect_metal_compiler_owned_vendor_payload(
        compile_result,
        "Softmax",
        "metal/vendor/mps_softmax",
        "mps_softmax",
        GfxAppleMpsVendorPrimitiveKind::Softmax,
        2u,
        1u,
        1u,
        1u);
#endif
}

TEST(GpuBackendBaseTest, MetalCompilerBundleOwnsMpsPool2DVendorPayload) {
#if GFX_BACKEND_METAL_AVAILABLE
    auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{1, 4, 16, 16});
    auto pool = std::make_shared<ov::op::v1::MaxPool>(
        input,
        ov::Strides{2, 2},
        ov::Shape{0, 0},
        ov::Shape{0, 0},
        ov::Shape{2, 2},
        ov::op::RoundingType::FLOOR);
    auto result = std::make_shared<ov::op::v0::Result>(pool);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{input});

    const auto compile_result = compile_metal_without_graph_pipeline(model);
    expect_metal_compiler_owned_vendor_payload(
        compile_result,
        "MaxPool",
        "metal/vendor/mps_pool2d",
        "mps_pool2d",
        GfxAppleMpsVendorPrimitiveKind::Pool2D,
        3u,
        1u,
        1u,
        1u);
#endif
}

TEST(GpuBackendBaseTest, MetalCompilerKeepsPool2DOffMpsWhenImageContractInvalid) {
#if GFX_BACKEND_METAL_AVAILABLE
    auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{1, 3, 16, 16});
    auto pool = std::make_shared<ov::op::v1::MaxPool>(
        input,
        ov::Strides{2, 2},
        ov::Shape{0, 0},
        ov::Shape{0, 0},
        ov::Shape{2, 2},
        ov::op::RoundingType::FLOOR);
    auto result = std::make_shared<ov::op::v0::Result>(pool);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{input});

    const auto compile_result = compile_metal_without_graph_pipeline(model);
    ASSERT_TRUE(compile_result.supported());
    const auto artifact_it = std::find_if(
        compile_result.executable.artifact_descriptors.begin(),
        compile_result.executable.artifact_descriptors.end(),
        [](const compiler::KernelArtifactDescriptor& artifact) {
            return artifact.kernel.op_family == "MaxPool";
        });
    ASSERT_NE(artifact_it, compile_result.executable.artifact_descriptors.end());
    EXPECT_NE(artifact_it->kernel.origin,
              compiler::KernelArtifactOrigin::VendorPrimitive);
    EXPECT_NE(artifact_it->payload_kind,
              compiler::KernelArtifactPayloadKind::VendorDescriptor);
    EXPECT_NE(artifact_it->kernel.kernel_id, "metal/vendor/mps_pool2d");
#endif
}

TEST(GpuBackendBaseTest, MetalCompilerBundleOwnsMpsResize2DVendorPayload) {
#if GFX_BACKEND_METAL_AVAILABLE
    auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{1, 4, 16, 16});
    auto output_shape = ov::op::v0::Constant::create(
        ov::element::i64,
        ov::Shape{2},
        std::vector<int64_t>{32, 32});
    ov::op::v0::Interpolate::Attributes attrs;
    attrs.axes = ov::AxisSet{2, 3};
    attrs.mode = "linear";
    attrs.align_corners = false;
    auto interpolate =
        std::make_shared<ov::op::v0::Interpolate>(input, output_shape, attrs);
    auto result = std::make_shared<ov::op::v0::Result>(interpolate);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{input});

    const auto compile_result = compile_metal_without_graph_pipeline(model);
    expect_metal_compiler_owned_vendor_payload(
        compile_result,
        "Interpolate",
        "metal/vendor/mps_resize2d",
        "mps_resize2d",
        GfxAppleMpsVendorPrimitiveKind::Resize2D,
        2u,
        1u,
        1u,
        1u);
#endif
}

TEST(GpuBackendBaseTest, MetalCompilerKeepsNearestResize2DOffMps) {
#if GFX_BACKEND_METAL_AVAILABLE
    auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{1, 4, 16, 16});
    auto output_shape = ov::op::v0::Constant::create(
        ov::element::i64,
        ov::Shape{2},
        std::vector<int64_t>{32, 32});
    ov::op::v0::Interpolate::Attributes attrs;
    attrs.axes = ov::AxisSet{2, 3};
    attrs.mode = "nearest";
    attrs.align_corners = false;
    auto interpolate =
        std::make_shared<ov::op::v0::Interpolate>(input, output_shape, attrs);
    auto result = std::make_shared<ov::op::v0::Result>(interpolate);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{input});

    const auto compile_result = compile_metal_without_graph_pipeline(model);
    ASSERT_TRUE(compile_result.supported());
    const auto artifact_it = std::find_if(
        compile_result.executable.artifact_descriptors.begin(),
        compile_result.executable.artifact_descriptors.end(),
        [](const compiler::KernelArtifactDescriptor& artifact) {
            return artifact.kernel.op_family == "Interpolate";
        });
    ASSERT_NE(artifact_it, compile_result.executable.artifact_descriptors.end());
    EXPECT_NE(artifact_it->kernel.origin,
              compiler::KernelArtifactOrigin::VendorPrimitive);
    EXPECT_NE(artifact_it->payload_kind,
              compiler::KernelArtifactPayloadKind::VendorDescriptor);
    EXPECT_NE(artifact_it->kernel.kernel_id, "metal/vendor/mps_resize2d");
#endif
}

TEST(GpuBackendBaseTest, MetalCompilerBundleOwnsMpsGraphSdpaVendorPayload) {
#if GFX_BACKEND_METAL_AVAILABLE
    auto query = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{1, 2, 3, 4});
    auto key = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{1, 2, 5, 4});
    auto value = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{1, 2, 5, 4});
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
        query,
        key,
        value,
        false);
    auto result = std::make_shared<ov::op::v0::Result>(sdpa);
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{result},
        ov::ParameterVector{query, key, value});

    const auto compile_result = compile_metal_without_graph_pipeline(model);
    expect_metal_compiler_owned_vendor_payload(
        compile_result,
        "ScaledDotProductAttention",
        "metal/vendor/mpsgraph_sdpa",
        "mps_sdpa",
        GfxAppleMpsVendorPrimitiveKind::Sdpa,
        4u,
        1u,
        3u,
        1u);
#endif
}

TEST(GpuBackendBaseTest, MetalStageFactoryConsumesMpsVendorDescriptorPayload) {
#if GFX_BACKEND_METAL_AVAILABLE
    auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{1, 8});
    auto softmax = std::make_shared<ov::op::v8::Softmax>(input, 1);
    auto result = std::make_shared<ov::op::v0::Result>(softmax);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{input});

    const auto compile_result = compile_metal_without_graph_pipeline(model);
    ASSERT_TRUE(compile_result.supported());
    const auto runtime_descriptor =
        RuntimeExecutableDescriptorBuilder{}.build(compile_result.executable);
    ASSERT_TRUE(runtime_descriptor.verify(compile_result.executable).valid());
    const auto stage_it = std::find_if(
        runtime_descriptor.stages.begin(),
        runtime_descriptor.stages.end(),
        [](const RuntimeStageExecutableDescriptor& stage) {
            return stage.op_family == "Softmax";
        });
    ASSERT_NE(stage_it, runtime_descriptor.stages.end());
    ASSERT_EQ(stage_it->payload_kind,
              compiler::KernelArtifactPayloadKind::VendorDescriptor);

    auto stage = create_metal_stage(softmax, nullptr, nullptr, &*stage_it);
    ASSERT_TRUE(stage);
    EXPECT_EQ(stage->type(), "MpsrtVendorPrimitive");
    EXPECT_EQ(stage->name(), softmax->get_friendly_name());
#endif
}

TEST(GpuBackendBaseTest, MetalCompilerBundleOwnsCausalSdpaMslPayload) {
#if GFX_BACKEND_METAL_AVAILABLE
    auto query = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f16,
        ov::Shape{1, 2, 3, 4});
    auto key = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f16,
        ov::Shape{1, 2, 5, 4});
    auto value = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f16,
        ov::Shape{1, 2, 5, 4});
    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64,
        ov::Shape{1, 5});
    auto cache_positions = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64,
        ov::Shape{3});
    auto scale = ov::op::v0::Constant::create(ov::element::f32,
                                              ov::Shape{},
                                              {0.5f});
    auto sdpa = std::make_shared<ov::gfx_plugin::op::GfxSDPAWithCausalMask>(
        ov::OutputVector{query,
                         key,
                         value,
                         attention_mask,
                         cache_positions,
                         scale});
    auto result = std::make_shared<ov::op::v0::Result>(sdpa);
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{result},
        ov::ParameterVector{query,
                            key,
                            value,
                            attention_mask,
                            cache_positions});

    const auto compile_result = compile_metal_without_graph_pipeline(model);
    expect_metal_compiler_owned_msl_payload(
        compile_result,
        "GfxSDPAWithCausalMask",
        "metal/generated/sdpa_causal_mask",
        "masked_softmax_attention",
        7u,
        1u,
        "cache_positions");
#endif
}

TEST(GpuBackendBaseTest, KernelSourcePayloadUsesOneContractForOpenClAndMsl) {
    const auto& opencl_source = opencl_baseline_binary_f32_kernel_source();
    ASSERT_TRUE(gfx_kernel_source_valid(opencl_source));
    EXPECT_EQ(gfx_kernel_source_payload_kind(opencl_source.source_language),
              compiler::KernelArtifactPayloadKind::OpenClSource);

    GfxKernelSourcePayload opencl_payload(opencl_source);
    EXPECT_TRUE(opencl_payload.valid());
    EXPECT_EQ(opencl_payload.payload_kind(),
              compiler::KernelArtifactPayloadKind::OpenClSource);
    EXPECT_EQ(opencl_payload.backend_domain(), "opencl");
    EXPECT_EQ(opencl_payload.source_id(), "opencl/baseline/eltwise_binary_f32");
    EXPECT_EQ(opencl_payload.entry_point(), "gfx_opencl_baseline_binary_f32");

#if GFX_BACKEND_METAL_AVAILABLE
    const auto& msl_source = mpsrt_image_bridge_kernel_source(
        MpsrtImageBridgeKernelKind::BufferToImageF32);
    ASSERT_TRUE(gfx_kernel_source_valid(msl_source));
    EXPECT_EQ(gfx_kernel_source_payload_kind(msl_source.source_language),
              compiler::KernelArtifactPayloadKind::MslSource);

    GfxKernelSourcePayload msl_payload(msl_source);
    EXPECT_TRUE(msl_payload.valid());
    EXPECT_EQ(msl_payload.payload_kind(),
              compiler::KernelArtifactPayloadKind::MslSource);
    EXPECT_EQ(msl_payload.backend_domain(), "apple_msl");
    EXPECT_EQ(msl_payload.source_id(), "mpsrt_bridge_buffer_to_image_f32");
    EXPECT_EQ(msl_payload.entry_point(), "gfx_mpsrt_buffer_to_image_f32");

    const auto& topk_source =
        mpsrt_topk_stable_i64_indices_kernel_source(MpsrtTopKValueType::F32);
    GfxKernelSourcePayload topk_payload(topk_source);
    EXPECT_TRUE(topk_payload.valid());
    EXPECT_EQ(topk_payload.payload_kind(),
              compiler::KernelArtifactPayloadKind::MslSource);
    EXPECT_EQ(topk_payload.backend_domain(), "apple_msl");
#endif
}

TEST(GpuBackendBaseTest, KernelBindingPlanValidatesDenseArgs) {
    FakeCompiledKernel kernel(/*arg_count=*/2);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x1);
    a.size = 16;
    GpuBuffer b{};
    b.buffer = reinterpret_cast<GpuBufferHandle>(0x2);
    b.size = 16;

    std::vector<KernelArg> args = {make_buffer_arg(0, a), make_buffer_arg(1, b)};
    EXPECT_EQ(kernel.resolve(args, "FakeKernel"), 2u);
}

TEST(GpuBackendBaseTest, KernelBindingPlanRejectsMismatchedArgCount) {
    FakeCompiledKernel kernel(/*arg_count=*/2);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x1);
    a.size = 16;

    std::vector<KernelArg> args = {make_buffer_arg(0, a)};
    EXPECT_THROW(kernel.resolve(args, "FakeKernel"), ov::Exception);
}

TEST(GpuBackendBaseTest, ForkReusesResolvedBindingPlan) {
    FakeCompiledKernel kernel;
    kernel.set_args_count(3);

    auto forked = kernel.fork();
    ASSERT_TRUE(forked);
    EXPECT_EQ(kernel.args_count(), 3u);
    EXPECT_EQ(forked->args_count(), 3u);
}

TEST(GpuBackendBaseTest, MaterializeKernelBindingTableProducesDenseOrderedBindings) {
    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    GpuBuffer b{};
    b.buffer = reinterpret_cast<GpuBufferHandle>(0x20);
    b.size = 128;

    std::vector<KernelArg> args;
    args.push_back(make_buffer_arg(1, b, 8));
    args.push_back(make_buffer_arg(0, a, 4));

    const auto table = materialize_kernel_binding_table(args, "FakeKernel");
    ASSERT_EQ(table.buffers.size(), 2u);
    EXPECT_EQ(table.buffers[0].buffer.buffer, a.buffer);
    EXPECT_EQ(table.buffers[0].offset, 4u);
    EXPECT_EQ(table.buffers[1].buffer.buffer, b.buffer);
    EXPECT_EQ(table.buffers[1].offset, 8u);
}

TEST(GpuBackendBaseTest, KernelBindingPlanMaterializesFixedAbiBindings) {
    FakeCompiledKernel kernel(/*arg_count=*/2);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    GpuBuffer b{};
    b.buffer = reinterpret_cast<GpuBufferHandle>(0x20);
    b.size = 128;

    std::vector<KernelArg> args = {make_buffer_arg(0, a, 4), make_buffer_arg(1, b, 8)};
    const auto table = kernel.bindings(args, "FakeKernel");
    ASSERT_EQ(table.buffers.size(), 2u);
    EXPECT_EQ(table.buffers[0].buffer.buffer, a.buffer);
    EXPECT_EQ(table.buffers[0].offset, 4u);
    EXPECT_EQ(table.buffers[1].buffer.buffer, b.buffer);
    EXPECT_EQ(table.buffers[1].offset, 8u);
}

TEST(GpuBackendBaseTest, KernelBindingTableUsesStableAllocationIdentityWhenAvailable) {
    KernelBindingTable first;
    KernelBindingTable second;

    first.buffers.resize(1);
    second.buffers.resize(1);
    first.buffers[0].buffer.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    second.buffers[0].buffer.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    first.buffers[0].buffer.size = 64;
    second.buffers[0].buffer.size = 64;
    first.buffers[0].buffer.allocation_uid = 1;
    second.buffers[0].buffer.allocation_uid = 2;

    EXPECT_FALSE(first == second);
    EXPECT_NE(KernelBindingTableHash{}(first), KernelBindingTableHash{}(second));
}

TEST(GpuBackendBaseTest, PreparedBindingsAreReusedAcrossForkedKernels) {
    FakeCompiledKernel kernel(/*arg_count=*/1);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    a.allocation_uid = 7;

    std::vector<KernelArg> args = {make_buffer_arg(0, a, 4)};
    auto first = kernel.prepared(args, "FakeKernel");
    ASSERT_TRUE(first);
    EXPECT_EQ(kernel.prepared_binding_create_count, 1u);

    auto forked = kernel.fork();
    ASSERT_TRUE(forked);
    auto* forked_kernel = dynamic_cast<FakeCompiledKernel*>(forked.get());
    ASSERT_NE(forked_kernel, nullptr);

    auto second = forked_kernel->prepared(args, "FakeKernel");
    ASSERT_TRUE(second);
    EXPECT_EQ(second.get(), first.get());
    EXPECT_EQ(kernel.prepared_binding_create_count, 1u);
    EXPECT_EQ(forked_kernel->prepared_binding_create_count, 0u);
}

TEST(GpuBackendBaseTest, PreparedBindingsAreReusedAcrossDistinctKernelsSharingRegistryCache) {
    auto binding_plan = std::make_shared<KernelBindingPlan>(1);
    auto shared_cache = acquire_shared_prepared_binding_cache(GpuBackend::Metal,
                                                              /*device=*/0x1234,
                                                              /*arg_count=*/1);

    FakeCompiledKernel first(binding_plan, shared_cache);
    FakeCompiledKernel second(binding_plan, shared_cache);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    a.allocation_uid = 7;

    std::vector<KernelArg> args = {make_buffer_arg(0, a, 4)};
    auto first_prepared = first.prepared(args, "FakeKernel");
    auto second_prepared = second.prepared(args, "FakeKernel");

    ASSERT_TRUE(first_prepared);
    ASSERT_TRUE(second_prepared);
    EXPECT_EQ(first_prepared.get(), second_prepared.get());
    EXPECT_EQ(first.prepared_binding_create_count, 1u);
    EXPECT_EQ(second.prepared_binding_create_count, 0u);
}

TEST(GpuBackendBaseTest, PreparedBindingsStayAliveInSharedCacheAcrossLookups) {
    FakeCompiledKernel kernel(/*arg_count=*/1);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    a.allocation_uid = 7;

    std::vector<KernelArg> args = {make_buffer_arg(0, a, 4)};
    auto first = kernel.prepared(args, "FakeKernel");
    ASSERT_TRUE(first);
    std::weak_ptr<const PreparedKernelBindings> weak = first;
    first.reset();

    EXPECT_FALSE(weak.expired());

    auto second = kernel.prepared(args, "FakeKernel");
    ASSERT_TRUE(second);
    ASSERT_FALSE(weak.expired());
    EXPECT_EQ(second.get(), weak.lock().get());
    EXPECT_EQ(kernel.prepared_binding_create_count, 1u);
}

TEST(GpuBackendBaseTest, PreparedBindingsReuseBackendStateForSameSchemaKey) {
    FakeCompiledKernel kernel(/*arg_count=*/1);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    a.allocation_uid = 7;

    std::vector<KernelArg> args = {make_buffer_arg(0, a, 4)};
    auto prepared = kernel.prepared(args, "FakeKernel");
    ASSERT_TRUE(prepared);

    size_t create_count = 0;
    auto first = prepared->get_or_create_backend_state<FakeBackendState>(
        /*state_key=*/0x100,
        [&]() {
            ++create_count;
            return std::make_shared<FakeBackendState>(17);
        });
    auto second = prepared->get_or_create_backend_state<FakeBackendState>(
        /*state_key=*/0x100,
        [&]() {
            ++create_count;
            return std::make_shared<FakeBackendState>(19);
        });

    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_EQ(first.get(), second.get());
    EXPECT_EQ(first->value, 17);
    EXPECT_EQ(create_count, 1u);
}

TEST(GpuBackendBaseTest, PreparedBindingsKeepBackendStateAliveAcrossTransientHandles) {
    FakeCompiledKernel kernel(/*arg_count=*/1);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    a.allocation_uid = 7;

    std::vector<KernelArg> args = {make_buffer_arg(0, a, 4)};
    auto prepared = kernel.prepared(args, "FakeKernel");
    ASSERT_TRUE(prepared);

    auto state = prepared->get_or_create_backend_state<FakeBackendState>(
        /*state_key=*/0x100,
        [&]() { return std::make_shared<FakeBackendState>(17); });
    ASSERT_TRUE(state);
    std::weak_ptr<FakeBackendState> weak_state = state;
    state.reset();
    prepared.reset();

    EXPECT_FALSE(weak_state.expired());

    auto prepared_again = kernel.prepared(args, "FakeKernel");
    ASSERT_TRUE(prepared_again);
    auto state_again = prepared_again->get_or_create_backend_state<FakeBackendState>(
        /*state_key=*/0x100,
        [&]() { return std::make_shared<FakeBackendState>(23); });

    ASSERT_TRUE(state_again);
    ASSERT_FALSE(weak_state.expired());
    EXPECT_EQ(state_again.get(), weak_state.lock().get());
    EXPECT_EQ(state_again->value, 17);
}

TEST(GpuBackendBaseTest, PreparedBindingsKeepBackendStateSeparatedPerSchemaKey) {
    FakeCompiledKernel kernel(/*arg_count=*/1);

    GpuBuffer a{};
    a.buffer = reinterpret_cast<GpuBufferHandle>(0x10);
    a.size = 64;
    a.allocation_uid = 7;

    std::vector<KernelArg> args = {make_buffer_arg(0, a, 4)};
    auto prepared = kernel.prepared(args, "FakeKernel");
    ASSERT_TRUE(prepared);

    auto first = prepared->get_or_create_backend_state<FakeBackendState>(
        /*state_key=*/0x100,
        [&]() { return std::make_shared<FakeBackendState>(17); });
    auto second = prepared->get_or_create_backend_state<FakeBackendState>(
        /*state_key=*/0x200,
        [&]() { return std::make_shared<FakeBackendState>(23); });

    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_NE(first.get(), second.get());
    EXPECT_EQ(first->value, 17);
    EXPECT_EQ(second->value, 23);
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
