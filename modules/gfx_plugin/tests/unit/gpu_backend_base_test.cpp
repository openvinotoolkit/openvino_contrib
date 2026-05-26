// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>

#include "backends/metal/compiler/metal_operation_support.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "compiler/backend_registry.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/gfx_compiler_service.hpp"
#include "compiler/manifest.hpp"
#include "compiler/operation_support.hpp"
#include "kernel_ir/gfx_kernel_source.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "kernel_ir/opencl_kernels/binary_f32_kernel.hpp"
#include "kernel_ir/metal_kernels/mpsrt_image_bridge_kernels.hpp"
#include "kernel_ir/metal_kernels/mpsrt_topk_kernels.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gpu_backend_base.hpp"

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
    ASSERT_TRUE(compile_result.supported());
    ASSERT_TRUE(compile_result.executable.verify().valid());

    const auto artifact_it = std::find_if(
        compile_result.executable.artifact_descriptors.begin(),
        compile_result.executable.artifact_descriptors.end(),
        [](const compiler::KernelArtifactDescriptor& artifact) {
            return artifact.kernel.op_family == "ShapeOf";
        });
    ASSERT_NE(artifact_it, compile_result.executable.artifact_descriptors.end());
    EXPECT_EQ(artifact_it->kernel.origin,
              compiler::KernelArtifactOrigin::Generated);
    EXPECT_EQ(artifact_it->payload_kind,
              compiler::KernelArtifactPayloadKind::MslSource);
    EXPECT_EQ(artifact_it->kernel.kernel_id, "metal/generated/shapeof");
    EXPECT_EQ(artifact_it->entry_point, "shapeof_kernel");
    EXPECT_EQ(artifact_it->abi_arg_count, 4u);
    EXPECT_EQ(artifact_it->abi_output_arg_count, 1u);

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
    EXPECT_NE(std::string(source_payload->source().source).find("shapeof_kernel"),
              std::string::npos);

    const auto runtime_descriptor =
        RuntimeExecutableDescriptorBuilder{}.build(compile_result.executable);
    ASSERT_TRUE(runtime_descriptor.verify(compile_result.executable).valid());
    const auto stage_it = std::find_if(
        runtime_descriptor.stages.begin(),
        runtime_descriptor.stages.end(),
        [](const RuntimeStageExecutableDescriptor& stage) {
            return stage.op_family == "ShapeOf";
        });
    ASSERT_NE(stage_it, runtime_descriptor.stages.end());
    EXPECT_EQ(stage_it->payload, payload);
    EXPECT_EQ(stage_it->payload_kind,
              compiler::KernelArtifactPayloadKind::MslSource);
    EXPECT_EQ(stage_it->abi_arg_count, artifact_it->abi_arg_count);
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
