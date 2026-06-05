// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backends/metal/compiler/apple_vendor_descriptors.hpp"
#include "backends/metal/compiler/metal_operation_support.hpp"
#include "backends/opencl/compiler/opencl_backend_module.hpp"
#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "common/gpu_backend.hpp"
#include "common/gpu_device_profile.hpp"
#include "common/gpu_parallelism_profile.hpp"
#include "compiler/backend_config.hpp"
#include "compiler/cache_envelope.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/gfx_compiler_service.hpp"
#include "compiler/memory_plan.hpp"
#include "compiler/pipeline_stage_builder.hpp"
#include "compiler/pipeline_stage_fusion.hpp"
#include "compiler/pipeline_stage_plan.hpp"
#include "compiler/runtime_executable_descriptor_builder.hpp"
#include "compiler/static_backend_module.hpp"
#include "compiler/tensor_layout.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "runtime/backend_request_state.hpp"
#include "runtime/backend_runtime.hpp"
#include "runtime/backend_runtime_provider.hpp"
#include "runtime/backend_stage_factory.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/execution_dispatcher.hpp"
#include "runtime/gpu_buffer.hpp"
#include "runtime/gpu_memory_ops.hpp"
#include "runtime/kernel_launch_plan.hpp"
#include "runtime/pipeline_stage_materializer.hpp"
#include "runtime/stateful_stage.hpp"
#include "runtime/view_only_stage.hpp"
#include "runtime/runtime_session.hpp"
#include "transforms/pipeline.hpp"
#include "unit/gfx_backend_contracts.hpp"

using ov::gfx_plugin::compiler::KernelUnitKind;
using ov::gfx_plugin::compiler::LoweringRouteKind;

namespace ov {
namespace gfx_plugin {
namespace {

class GfxBackendArchitectureContractTest : public ::testing::Test {
protected:
  test::BackendContractCatalog backend_catalog;
  test::ModelContractFactory models;
};

static_assert(
    std::is_base_of_v<BackendStageFactory, BackendState>,
    "Backend runtime state must expose stage materialization through the "
    "runtime-facing BackendStageFactory interface");

static_assert(
    std::is_same_v<decltype(&BackendState::create_stage),
                   std::unique_ptr<GpuStage> (BackendState::*)(
                       const RuntimeStageMaterializationContext &) const>,
    "BackendState must materialize stages through the unified compiler-owned "
    "runtime descriptor context");

static_assert(
    std::is_same_v<
        decltype(&GpuStageFactory::create),
        std::unique_ptr<GpuStage> (*)(
            const RuntimeStageMaterializationContext &, GpuBackend, void *,
            void *)>,
    "GpuStageFactory must require the unified compiler-owned stage "
    "materialization context");

static_assert(std::is_same_v<decltype(&RuntimeSession::prepare_stage),
                             PreparedKernelExecutable (RuntimeSession::*)(
                                 size_t, GpuStage &, GpuBufferManager *,
                                 ResourceBindingTable) const>,
              "RuntimeSession must own request-time executable preparation");

static_assert(
    std::is_same_v<decltype(&create_stateful_stage),
                   std::unique_ptr<GpuStage> (*)(
                       const RuntimeStageExecutableDescriptor &)>,
    "Shared stateful runtime stage materialization must use the compiler-owned "
    "stage descriptor, not an ov::Node");

static_assert(
    std::is_same_v<decltype(&create_view_only_stage),
                   std::unique_ptr<GpuStage> (*)(
                       const RuntimeStageExecutableDescriptor &)>,
    "Shared view-only runtime stage materialization must use the compiler-owned "
    "stage descriptor, not an ov::Node");

static_assert(
    std::is_same_v<decltype(&PipelineStageMaterializer::create_stage),
                   std::unique_ptr<GpuStage> (PipelineStageMaterializer::*)(
                       const std::shared_ptr<const ov::Node> &) const>,
    "PipelineStageMaterializer must own descriptor-backed stage prototype "
    "materialization for CompiledModel");

static_assert(
    std::is_same_v<decltype(&PipelineStageMaterializer::stage_index_for),
                   size_t (PipelineStageMaterializer::*)(
                       const std::shared_ptr<const ov::Node> &) const>,
    "PipelineStageMaterializer must expose descriptor-owned runtime stage "
    "indices without fallback repair");

static_assert(
    std::is_same_v<
        decltype(PipelineStageRuntimeMaterializationRequest::runtime_plan),
        const PipelineStageRuntimePlan *>,
    "Runtime materializer must consume a runtime-facing stage plan instead of "
    "the compiler build result");

static_assert(
    std::is_same_v<decltype(&compiler::build_pipeline_stage_runtime_plan),
                   std::shared_ptr<const PipelineStageRuntimePlan> (*)(
                       const compiler::PipelineStageBuildRequest &)>,
    "Compiler pipeline stage builder must return an immutable runtime-facing "
    "stage plan instead of exposing mutable compiler build state");

static_assert(
    std::is_same_v<decltype(&materialize_pipeline_stage_descriptors),
                   std::vector<PipelineStageDesc> (*)(
                       const PipelineStageRuntimeMaterializationRequest &)>,
    "Runtime materializer must be the only shared path that turns compiler "
    "stage plans into runtime PipelineStageDesc objects");

static_assert(
    std::is_same_v<decltype(&transforms::run_pipeline),
                   std::shared_ptr<const ov::Model> (*)(
                       const std::shared_ptr<const ov::Model> &,
                       const transforms::PipelineOptions &)>,
    "Common graph pipeline must be configured by backend-owned options, "
    "not by a direct GpuBackend branch key");

static_assert(
    !std::is_invocable_v<decltype(&transforms::run_pipeline),
                         const std::shared_ptr<const ov::Model> &, GpuBackend>,
    "Common graph pipeline must not accept GpuBackend directly");

static_assert(
    !std::is_invocable_v<decltype(&compiler::BackendRegistry::resolve),
                         const compiler::BackendRegistry *, GpuBackend>,
    "Compiler backend registry must require exact BackendTarget resolution, "
    "not backend enum resolution");

static_assert(
    std::is_same_v<decltype(BackendRuntimeProvider::create_state),
                   std::unique_ptr<BackendState> (*)(
                       const compiler::BackendTarget&,
                       const ov::AnyMap&,
                       const ov::SoPtr<ov::IRemoteContext>&)>,
    "BackendRuntimeProvider must create runtime state for a concrete "
    "compiler BackendTarget, not for a backend enum alone");

static_assert(
    std::is_same_v<decltype(BackendRuntimeProvider::execute_infer),
                   void (*)(InferRequest &,
                            const std::shared_ptr<const CompiledModel> &)>,
    "BackendRuntimeProvider must expose one narrow infer execution callback "
    "instead of requiring common InferRequest backend switches");

static_assert(
    !std::is_invocable_v<decltype(&create_backend_state),
                         GpuBackend,
                         const ov::AnyMap&,
                         const ov::SoPtr<ov::IRemoteContext>&>,
    "Runtime state creation must not accept GpuBackend-only identity");

static_assert(
    !std::is_invocable_v<decltype(&execute_backend_infer),
                         GpuBackend,
                         InferRequest&,
                         const std::shared_ptr<const CompiledModel>&>,
    "Runtime infer dispatch must be entered through a compiler BackendTarget");

static_assert(
    std::is_same_v<decltype(&compiler::BackendCapabilities::precision),
                   const compiler::PrecisionCapabilities &(
                       compiler::BackendCapabilities::*)() const noexcept>,
    "Public precision capabilities must be owned by immutable "
    "compiler backend capabilities, not by runtime helpers");

static_assert(
    std::is_same_v<decltype(&compiler::BackendCapabilities::artifact_formats),
                   const compiler::ArtifactFormatCapabilities &(
                       compiler::BackendCapabilities::*)() const noexcept>,
    "Public artifact capabilities must be owned by immutable "
    "compiler backend capabilities, not by runtime helpers");

bool has_diagnostic_containing(const std::vector<std::string> &diagnostics,
                               std::string_view needle) {
  for (const auto &diagnostic : diagnostics) {
    if (diagnostic.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool target_has_feature(const compiler::BackendTarget &target,
                        std::string_view feature) {
  for (const auto &candidate : target.feature_bits()) {
    if (candidate == feature) {
      return true;
    }
  }
  return false;
}

GpuTensor make_test_launch_plan_tensor(uint64_t allocation_uid) {
  GpuTensor tensor;
  tensor.buf.buffer = reinterpret_cast<GpuBufferHandle>(
      static_cast<uintptr_t>(0x1000u + allocation_uid));
  tensor.buf.size = 64;
  tensor.buf.allocation_uid = allocation_uid;
  tensor.shape = ov::Shape{16};
  tensor.expected_type = ov::element::f32;
  return tensor;
}

class UnitVendorPayload final : public KernelArtifactPayload {
public:
  KernelArtifactPayloadKind payload_kind() const noexcept override {
    return KernelArtifactPayloadKind::VendorDescriptor;
  }

  std::string_view backend_domain() const noexcept override {
    return kBackendMetal;
  }

  std::string_view source_id() const noexcept override {
    return "unit_vendor_descriptor";
  }

  std::string_view entry_point() const noexcept override {
    return "unit_vendor_entry";
  }

  bool valid() const noexcept override {
    return true;
  }
};

class UnitBackendStageFactory final : public BackendStageFactory {
public:
  GpuBackend backend() const override {
    return GpuBackend::Metal;
  }

  std::unique_ptr<GpuStage>
  create_stage(const RuntimeStageMaterializationContext &) const override {
    return nullptr;
  }
};

RuntimeTensorBindingContract make_runtime_binding(std::string logical_name,
                                                  std::string region_id,
                                                  std::string role) {
  RuntimeTensorBindingContract binding;
  binding.logical_name = std::move(logical_name);
  binding.memory_region_id = std::move(region_id);
  binding.role = std::move(role);
  binding.element_type = "f32";
  binding.partial_shape = "{1}";
  binding.layout = "logical";
  binding.storage_kind = "device_buffer";
  binding.lifetime_class = "unit_lifetime";
  binding.alias_group = binding.memory_region_id;
  return binding;
}

RuntimeStageExecutableDescriptor make_materializer_base_descriptor(
    const std::shared_ptr<ov::Node> &node) {
  RuntimeStageExecutableDescriptor descriptor;
  descriptor.stage_index = 0;
  descriptor.stage_record_key = 0x1234u;
  descriptor.artifact_descriptor_index = 0;
  descriptor.manifest_ref = "manifest://unit/relu";
  descriptor.abi_fingerprint = "abi://unit/relu";
  descriptor.artifact_key = "artifact://unit/relu";
  descriptor.backend_domain = kBackendMetal;
  descriptor.kernel_id = "unit/relu";
  descriptor.op_family = node->get_type_name();
  descriptor.stage_name = node->get_friendly_name();
  descriptor.origin = KernelArtifactOrigin::Generated;
  descriptor.payload_kind = KernelArtifactPayloadKind::MslSource;
  descriptor.entry_point = "unit_relu";
  descriptor.abi_arg_count = 1;
  descriptor.abi_output_arg_count = 1;
  descriptor.tensor_roles = {"TensorInput", "TensorOutput"};
  descriptor.input_bindings.push_back(make_runtime_binding(
      "parameter.output0", "compiler_input_region", "TensorInput"));
  descriptor.output_bindings.push_back(make_runtime_binding(
      "relu.output0", "compiler_output_region", "TensorOutput"));
  return descriptor;
}

PipelineStageMaterializationPlan make_vendor_materialization_plan(
    const std::shared_ptr<ov::Node> &input,
    const std::shared_ptr<ov::Node> &node,
    RuntimeStageExecutableDescriptor descriptor) {
  descriptor.payload_kind = KernelArtifactPayloadKind::VendorDescriptor;
  descriptor.payload = std::make_shared<UnitVendorPayload>();
  descriptor.artifact_key = "artifact://unit/vendor_attention";

  PipelineStageMaterializationPlan plan;
  plan.kind = PipelineStageMaterializationKind::VendorAttention;
  plan.io_plan.node = node;
  plan.io_plan.runtime_stage_index = 0;
  plan.io_plan.inputs.push_back({input, 0});

  PipelineStageOutputDesc output;
  output.shape = ov::Shape{1};
  output.type = ov::element::f32;
  output.source_node = node;
  output.source_port = 0;
  plan.io_plan.outputs.push_back(std::move(output));

  plan.vendor_attention.name = "unit_vendor_attention";
  plan.vendor_attention.descriptor = descriptor;
  plan.materialized_descriptor = std::move(descriptor);
  plan.materialized_descriptor_valid = true;
  return plan;
}

TEST_F(GfxBackendArchitectureContractTest,
       KernelLaunchPlanBindsManifestRolesForConstAndRuntimeParams) {
  const std::vector<GfxKernelBufferRole> roles = {
      GfxKernelBufferRole::TensorInput,   GfxKernelBufferRole::ConstTensor,
      GfxKernelBufferRole::TensorOutput,  GfxKernelBufferRole::ScalarParam,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
  };

  auto input = make_test_launch_plan_tensor(1);
  auto const_tensor = make_test_launch_plan_tensor(2);
  auto output = make_test_launch_plan_tensor(3);
  auto runtime0 = make_test_launch_plan_tensor(4);
  auto runtime1 = make_test_launch_plan_tensor(5);

  const std::vector<size_t> direct_input_indices = {2};
  const std::vector<uint32_t> scalar_values = {42u};
  std::vector<GpuTensor *> outputs = {&output};
  std::vector<GpuTensor *> const_tensors = {&const_tensor};
  std::vector<GpuTensor> runtime_params = {runtime0, runtime1};

  size_t resolved_input_index = 0;
  auto plan = build_role_ordered_kernel_launch_plan<uint32_t>(
      roles, direct_input_indices, scalar_values, outputs, const_tensors,
      runtime_params,
      [&](size_t input_idx) -> GpuTensor * {
        resolved_input_index = input_idx;
        return &input;
      },
      "unit_launch_plan");

  ASSERT_EQ(plan.args.size(), roles.size());
  EXPECT_TRUE(kernel_args_dense(plan.args));
  EXPECT_EQ(resolved_input_index, 2u);

  EXPECT_EQ(plan.args[0].kind, KernelArg::Kind::Buffer);
  EXPECT_EQ(plan.args[0].index, 0u);
  EXPECT_EQ(plan.args[0].buffer.allocation_uid, 1u);
  EXPECT_EQ(plan.args[1].buffer.allocation_uid, 2u);
  EXPECT_EQ(plan.args[2].buffer.allocation_uid, 3u);
  EXPECT_EQ(plan.args[3].kind, KernelArg::Kind::Bytes);
  EXPECT_EQ(plan.args[3].index, 3u);
  ASSERT_EQ(plan.args[3].byte_size, sizeof(uint32_t));
  uint32_t scalar_value = 0;
  std::memcpy(&scalar_value, plan.args[3].bytes, sizeof(scalar_value));
  EXPECT_EQ(scalar_value, 42u);
  EXPECT_EQ(plan.args[4].buffer.allocation_uid, 4u);
  EXPECT_EQ(plan.args[5].buffer.allocation_uid, 5u);
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeStageMaterializationContextUsesDescriptorIdentityWithoutNode) {
  RuntimeStageExecutableDescriptor descriptor;
  descriptor.op_family = "DescriptorOwnedOp";
  descriptor.stage_name = "descriptor_owned_stage";
  descriptor.manifest_ref = "manifest://unit/descriptor_owned_stage";
  descriptor.kernel_id = "unit/descriptor_owned_stage";

  RuntimeStageMaterializationContext context{
      std::shared_ptr<const ov::Node>{}, descriptor};

  EXPECT_EQ(context.op_type_name(), std::string("DescriptorOwnedOp"));
  EXPECT_EQ(context.op_friendly_name(),
            std::string("descriptor_owned_stage"));
  EXPECT_EQ(&context.require_descriptor(), &descriptor);
}

TEST_F(GfxBackendArchitectureContractTest,
       KernelLaunchPlanRejectsUnconsumedConstAndRuntimeParamBuffers) {
  const std::vector<GfxKernelBufferRole> roles = {
      GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::TensorOutput,
  };

  auto input = make_test_launch_plan_tensor(1);
  auto output = make_test_launch_plan_tensor(2);
  auto extra_const = make_test_launch_plan_tensor(3);
  auto extra_runtime = make_test_launch_plan_tensor(4);

  const std::vector<size_t> direct_input_indices = {0};
  const std::vector<uint32_t> scalar_values;
  std::vector<GpuTensor *> outputs = {&output};
  std::vector<GpuTensor *> const_tensors = {&extra_const};
  std::vector<GpuTensor> runtime_params = {extra_runtime};

  EXPECT_THROW((void)build_role_ordered_kernel_launch_plan<uint32_t>(
                   roles, direct_input_indices, scalar_values, outputs,
                   const_tensors, {},
                   [&](size_t) -> GpuTensor * { return &input; },
                   "unit_extra_const"),
               ov::Exception);
  EXPECT_THROW((void)build_role_ordered_kernel_launch_plan<uint32_t>(
                   roles, direct_input_indices, scalar_values, outputs, {},
                   runtime_params,
                   [&](size_t) -> GpuTensor * { return &input; },
                   "unit_extra_runtime"),
               ov::Exception);
}

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStageFusionContractRejectsRuntimeProbeOnlyRoutes) {
  compiler::PipelineStageFusionContract msl_multiply;
  msl_multiply.op_family = "Multiply";
  msl_multiply.payload_kind = compiler::KernelArtifactPayloadKind::MslSource;
  msl_multiply.element_type = ov::element::f32;

  EXPECT_TRUE(compiler::allow_stage_input_activation_fusion(
      msl_multiply, 0, ActivationKind::Relu));
  EXPECT_TRUE(compiler::allow_stage_input_activation_fusion(
      msl_multiply, 1, ActivationKind::Swish));
  EXPECT_FALSE(compiler::allow_stage_input_activation_fusion(
      msl_multiply, 2, ActivationKind::Relu));
  EXPECT_FALSE(compiler::allow_stage_input_activation_fusion(
      msl_multiply, 0, ActivationKind::Abs));

  auto integral_multiply = msl_multiply;
  integral_multiply.element_type = ov::element::i32;
  EXPECT_FALSE(compiler::allow_stage_input_activation_fusion(
      integral_multiply, 0, ActivationKind::Relu));

  auto opencl_multiply = msl_multiply;
  opencl_multiply.payload_kind =
      compiler::KernelArtifactPayloadKind::OpenClSource;
  EXPECT_FALSE(compiler::allow_stage_input_activation_fusion(
      opencl_multiply, 0, ActivationKind::Relu));

  compiler::PipelineStageFusionContract msl_rms;
  msl_rms.op_family = "RMS";
  msl_rms.payload_kind = compiler::KernelArtifactPayloadKind::MslSource;
  EXPECT_TRUE(compiler::allow_stage_residual_add_fusion(msl_rms));

  auto opencl_rms = msl_rms;
  opencl_rms.payload_kind = compiler::KernelArtifactPayloadKind::OpenClSource;
  EXPECT_FALSE(compiler::allow_stage_residual_add_fusion(opencl_rms));
}

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStagePlanBuilderMarksOutputsAndDeduplicatesAliases) {
  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto relu = std::make_shared<ov::op::v0::Relu>(parameter);
  auto result = std::make_shared<ov::op::v0::Result>(relu);
  ov::Model model(ov::ResultVector{result}, ov::ParameterVector{parameter},
                  "pipeline_stage_plan_contract");

  const auto model_outputs = compiler::collect_model_output_ports(model);
  compiler::PipelineStagePlanBuilder builder(model_outputs);
  auto plan = builder.make_stage_plan(relu, 7);

  ASSERT_EQ(plan.outputs.size(), 1u);
  EXPECT_EQ(plan.runtime_stage_index, 7u);
  EXPECT_TRUE(plan.outputs[0].is_model_output);
  EXPECT_EQ(plan.outputs[0].source_node.get(), relu.get());
  EXPECT_EQ(plan.outputs[0].source_port, 0u);

  builder.append_output_alias(plan, relu, 0, 0);
  builder.append_output_alias(plan, relu, 0, 0);
  ASSERT_EQ(plan.output_aliases.size(), 1u);
  EXPECT_EQ(plan.output_aliases[0].node.get(), relu.get());

  compiler::PipelineOutputAliasMap aliases;
  builder.record_output_alias(aliases, relu.get(), 0, 3);
  const auto remapped = builder.remap_input_link(aliases, relu, 0);
  EXPECT_EQ(remapped.node.get(), relu.get());
  EXPECT_EQ(remapped.port, 3u);

  auto fused = builder.make_fused_stage_plan(relu, 2, 9);
  builder.describe_output(fused, 1, relu, 0);
  ASSERT_EQ(fused.outputs.size(), 2u);
  EXPECT_EQ(fused.runtime_stage_index, 9u);
  EXPECT_TRUE(fused.outputs[1].is_model_output);
  EXPECT_EQ(fused.outputs[1].source_node.get(), relu.get());
}

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStagePlanBuilderOnlyDirectAliasesStatefulAssignWhenExclusive) {
  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto variable = std::make_shared<ov::op::util::Variable>(
      ov::op::util::VariableInfo{ov::Shape{1}, ov::element::f32, "variable0"});
  auto add = std::make_shared<ov::op::v1::Add>(parameter, parameter);
  auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
  auto unrelated_result = std::make_shared<ov::op::v0::Result>(parameter);
  ov::Model assign_only_model(
      ov::ResultVector{unrelated_result}, ov::SinkVector{assign},
      ov::ParameterVector{parameter}, "stateful_assign_direct_alias_safe");

  const auto assign_only_outputs =
      compiler::collect_model_output_ports(assign_only_model);
  compiler::PipelineStagePlanBuilder assign_only_builder(assign_only_outputs);
  const auto assign_only_plan = assign_only_builder.make_stage_plan(add, 0);
  ASSERT_EQ(assign_only_plan.outputs.size(), 1u);
  EXPECT_EQ(assign_only_plan.outputs[0].direct_stateful_assign_variable_id,
            "variable0");

  auto shared_parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto shared_variable = std::make_shared<ov::op::util::Variable>(
      ov::op::util::VariableInfo{ov::Shape{1}, ov::element::f32, "variable0"});
  auto shared_add =
      std::make_shared<ov::op::v1::Add>(shared_parameter, shared_parameter);
  auto shared_assign =
      std::make_shared<ov::op::v6::Assign>(shared_add, shared_variable);
  auto shared_result = std::make_shared<ov::op::v0::Result>(shared_add);
  ov::Model shared_consumer_model(ov::ResultVector{shared_result},
                                  ov::SinkVector{shared_assign},
                                  ov::ParameterVector{shared_parameter},
                                  "stateful_assign_direct_alias_unsafe");

  const auto shared_outputs =
      compiler::collect_model_output_ports(shared_consumer_model);
  compiler::PipelineStagePlanBuilder shared_builder(shared_outputs);
  const auto shared_plan = shared_builder.make_stage_plan(shared_add, 0);
  ASSERT_EQ(shared_plan.outputs.size(), 1u);
  EXPECT_TRUE(
      shared_plan.outputs[0].direct_stateful_assign_variable_id.empty());
}

TEST_F(GfxBackendArchitectureContractTest,
       ManifestCarriesDirectStatefulAssignPrebindShapeContract) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), 2});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), 2});
  auto concat = std::make_shared<ov::op::v0::Concat>(
      ov::OutputVector{lhs->output(0), rhs->output(0)}, 0);
  auto variable = std::make_shared<ov::op::util::Variable>(
      ov::op::util::VariableInfo{ov::PartialShape{ov::Dimension::dynamic(), 2},
                                 ov::element::f32, "variable0"});
  auto assign = std::make_shared<ov::op::v6::Assign>(concat, variable);
  ASSERT_NE(assign, nullptr);
  ASSERT_EQ(concat->output(0).get_target_inputs().size(), 1u);

  compiler::PlannedOperation op;
  op.source_node = concat;
  op.node_name = concat->get_friendly_name();
  op.type_name = concat->get_type_name();
  op.kernel_unit = compiler::KernelUnit::describe(
      LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
      "opencl", "metadata", true);
  op.layout = compiler::select_tensor_layout_plan("Concat", concat);
  op.profitability_score = 1.0;
  op.input_element_types = {"f32", "f32"};
  op.input_shapes = {"{?,2}", "{?,2}"};
  op.output_element_types = {"f32"};
  op.output_shapes = {"{?,2}"};

  compiler::LoweringPlan plan;
  plan.target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  plan.operations.push_back(std::move(op));

  const auto manifest = compiler::ManifestBuilder{}.build(plan);
  ASSERT_TRUE(manifest.valid());
  ASSERT_EQ(manifest.stages.size(), 1u);
  ASSERT_EQ(manifest.stages.front().outputs.size(), 1u);
  const auto &output = manifest.stages.front().outputs.front();
  EXPECT_EQ(output.stateful_prebind_variable_id, "variable0");
  EXPECT_EQ(output.stateful_prebind_shape_rule, "sum_inputs_along_axis");
  EXPECT_EQ(output.stateful_prebind_shape_axis, 0);

  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.valid());
  const auto runtime_descriptor =
      compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
  ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
  ASSERT_EQ(runtime_descriptor.stages.front().output_bindings.size(), 1u);
  const auto &runtime_output =
      runtime_descriptor.stages.front().output_bindings.front();
  EXPECT_EQ(runtime_output.stateful_prebind_variable_id, "variable0");
  EXPECT_EQ(runtime_output.stateful_prebind_shape_rule,
            "sum_inputs_along_axis");
  EXPECT_EQ(runtime_output.stateful_prebind_shape_axis, 0);
  EXPECT_TRUE(compiler::runtime_executable_descriptor_valid(runtime_descriptor,
                                                            executable));

  auto stale_runtime_descriptor = runtime_descriptor;
  stale_runtime_descriptor.stages.front()
      .output_bindings.front()
      .stateful_prebind_shape_rule = "stale_runtime_rule";
  const auto stale_verification =
      compiler::verify_runtime_executable_descriptor(stale_runtime_descriptor,
                                                     executable);
  EXPECT_FALSE(stale_verification.valid());
  EXPECT_TRUE(has_diagnostic_containing(stale_verification.diagnostics,
                                        "output binding drift"));
}

TEST_F(GfxBackendArchitectureContractTest,
       SharedStatefulAndViewOnlyStagesMaterializeFromDescriptorOnly) {
  RuntimeStageExecutableDescriptor assign_descriptor;
  assign_descriptor.manifest_ref = "manifest://unit/assign";
  assign_descriptor.kernel_id = "metadata";
  assign_descriptor.op_family = "Assign";
  assign_descriptor.stage_name = "unit_assign";
  assign_descriptor.stateful_effect = "assign";
  assign_descriptor.output_bindings.push_back(
      make_runtime_binding("assign.output0", "assign_output", "TensorOutput"));

  const auto assign_stage = create_stateful_stage(assign_descriptor);
  ASSERT_NE(assign_stage, nullptr);
  EXPECT_EQ(assign_stage->type(), std::string("Assign"));
  EXPECT_EQ(assign_stage->name(), std::string("assign.output0"));

  RuntimeStageExecutableDescriptor read_descriptor;
  read_descriptor.manifest_ref = "manifest://unit/read_value";
  read_descriptor.kernel_id = "metadata";
  read_descriptor.op_family = "ReadValue";
  read_descriptor.stage_name = "unit_read_value";
  read_descriptor.stateful_effect = "read_value";
  read_descriptor.output_bindings.push_back(make_runtime_binding(
      "read_value.output0", "read_value_output", "TensorOutput"));

  const auto read_stage = create_stateful_stage(read_descriptor);
  ASSERT_NE(read_stage, nullptr);
  EXPECT_EQ(read_stage->type(), std::string("ReadValue"));
  EXPECT_EQ(read_stage->name(), std::string("read_value.output0"));

  RuntimeStageExecutableDescriptor view_descriptor;
  view_descriptor.manifest_ref = "manifest://unit/view_only";
  view_descriptor.kernel_id = "metadata";
  view_descriptor.op_family = "Reshape";
  view_descriptor.stage_name = "unit_view_only";
  view_descriptor.origin = KernelArtifactOrigin::Metadata;
  view_descriptor.payload_kind = KernelArtifactPayloadKind::None;
  view_descriptor.tensor_view_only = true;
  view_descriptor.layout_contract = "view_only";
  view_descriptor.output_bindings.push_back(
      make_runtime_binding("reshape.output0", "reshape_output", "TensorOutput"));
  view_descriptor.output_bindings.front().partial_shape = "{2,3}";
  view_descriptor.output_bindings.front().element_type = "f32";

  auto view_stage = create_view_only_stage(view_descriptor);
  ASSERT_NE(view_stage, nullptr);
  EXPECT_EQ(view_stage->type(), std::string("Reshape"));
  EXPECT_EQ(view_stage->name(), std::string("reshape.output0"));

  GpuTensor input;
  input.buf.buffer = reinterpret_cast<GpuBufferHandle>(0x1234);
  input.buf.size = 24;
  input.buf.allocation_uid = 77;
  input.shape = ov::Shape{6};
  input.expected_type = ov::element::f32;
  GpuTensor output;

  view_stage->set_inputs({&input});
  view_stage->set_output(&output);
  view_stage->execute(nullptr);

  EXPECT_TRUE(same_gpu_allocation(input.buf, output.buf));
  EXPECT_FALSE(output.buf.owned);
  EXPECT_TRUE(output.buf.external);
  EXPECT_EQ(output.shape, (ov::Shape{2, 3}));
  EXPECT_EQ(output.expected_type, ov::element::f32);
}

compiler::TensorContract
make_tensor_contract(compiler::TensorContractRole role) {
  compiler::TensorContract contract;
  contract.logical_name =
      role == compiler::TensorContractRole::TensorInput ? "input0" : "output0";
  contract.memory_region_id = role == compiler::TensorContractRole::TensorInput
                                  ? "stage_0.input_0"
                                  : "stage_0.output_0";
  contract.role = role;
  contract.element_type = "f32";
  contract.partial_shape = "{1,3}";
  contract.layout = "logical";
  contract.storage_kind = "device_buffer";
  contract.lifetime_class = role == compiler::TensorContractRole::TensorInput
                                ? "producer_or_external"
                                : "stage_output";
  return contract;
}

compiler::MemoryRegion
make_memory_region_for_contract(const compiler::TensorContract &contract,
                                size_t stage_id) {
  compiler::MemoryRegion region;
  region.region_id = contract.memory_region_id;
  region.logical_tensor_name = contract.logical_name;
  region.kind = contract.role == compiler::TensorContractRole::TensorInput
                    ? compiler::MemoryRegionKind::ExternalTensor
                    : compiler::MemoryRegionKind::TransientTensor;
  region.element_type = contract.element_type;
  region.partial_shape = contract.partial_shape;
  region.layout = contract.layout;
  region.storage_kind = contract.storage_kind;
  region.alias_group =
      contract.role == compiler::TensorContractRole::TensorInput
          ? contract.memory_region_id
          : "stage_" + std::to_string(stage_id);
  region.lifetime = {0, stage_id};
  region.external_binding =
      contract.role == compiler::TensorContractRole::TensorInput;
  return region;
}

compiler::MemoryPlan
make_single_stage_memory_plan(const compiler::StageRecord &stage) {
  compiler::MemoryPlan plan;
  plan.schema_version = 1;
  for (const auto &input : stage.inputs) {
    auto region = make_memory_region_for_contract(input, stage.stage_id);
    compiler::AliasGroup group;
    group.group_id = region.alias_group;
    group.region_ids.push_back(region.region_id);
    plan.alias_groups.push_back(std::move(group));
    plan.regions.push_back(std::move(region));
  }
  compiler::TransientArena arena;
  arena.arena_id = "transient_device_buffer_arena";
  arena.storage_kind = "device_buffer";
  compiler::AliasGroup output_group;
  output_group.group_id = stage.memory.alias_group;
  for (const auto &output : stage.outputs) {
    auto region = make_memory_region_for_contract(output, stage.stage_id);
    output_group.region_ids.push_back(region.region_id);
    arena.region_ids.push_back(region.region_id);
    plan.regions.push_back(std::move(region));
  }
  if (!output_group.region_ids.empty()) {
    plan.alias_groups.push_back(std::move(output_group));
  }
  if (!arena.region_ids.empty()) {
    plan.transient_arenas.push_back(std::move(arena));
  }
  return plan;
}

compiler::ManifestBundle make_single_payload_route_manifest(
    LoweringRouteKind route_kind, std::string backend_domain,
    std::string kernel_unit_id, std::string kernel_unit_kind,
    std::string op_family = "PayloadRoute",
    bool requires_runtime_shape_args = false) {
  compiler::StageRecord stage;
  stage.stage_id = 0;
  stage.stable_record_key = 0x1234u;
  stage.source_node_name = op_family;
  stage.normalized_op_family = std::move(op_family);
  stage.execution_kind = route_kind;
  stage.backend_domain = std::move(backend_domain);
  stage.kernel_unit_id = std::move(kernel_unit_id);
  stage.kernel_unit_kind = std::move(kernel_unit_kind);
  stage.requires_runtime_shape_args = requires_runtime_shape_args;
  stage.inputs.push_back(
      make_tensor_contract(compiler::TensorContractRole::TensorInput));
  stage.outputs.push_back(
      make_tensor_contract(compiler::TensorContractRole::TensorOutput));
  stage.dispatch.execution_kind = stage.execution_kind;
  stage.dispatch.backend_domain = stage.backend_domain;
  stage.dispatch.kernel_unit_id = stage.kernel_unit_id;
  stage.dispatch.kernel_unit_kind = stage.kernel_unit_kind;
  stage.dispatch.dispatch_source = "manifest";
  stage.memory.alias_group = "stage_0";

  compiler::ManifestBundle manifest;
  manifest.schema_version = 2;
  manifest.target_fingerprint = stage.backend_domain + ":test-target";
  manifest.memory_plan = make_single_stage_memory_plan(stage);
  manifest.stages.push_back(std::move(stage));
  return manifest;
}

std::shared_ptr<ov::Model> make_relu_model() {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{2, 3});
  auto relu = std::make_shared<ov::op::v0::Relu>(input);
  auto result = std::make_shared<ov::op::v0::Result>(relu);
  return std::make_shared<ov::Model>(ov::ResultVector{result},
                                     ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> make_static_range_model() {
  auto start =
      ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
  auto stop =
      ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {4.0f});
  auto step =
      ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
  auto range =
      std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::f32);
  auto result = std::make_shared<ov::op::v0::Result>(range);
  return std::make_shared<ov::Model>(ov::ResultVector{result},
                                     ov::ParameterVector{});
}

compiler::CacheEnvelopeBuildOptions make_test_cache_options(
    const ov::Model &model,
    std::string backend_capabilities_fingerprint = "test-capabilities-v1",
    std::string backend_compiler_revision = "test-backend-compiler-v1",
    std::string driver_identity = "test-driver-v1") {
  compiler::CacheEnvelopeBuildOptions options;
  options.model_fingerprint = compiler::make_model_cache_fingerprint(model);
  options.backend_capabilities_fingerprint =
      std::move(backend_capabilities_fingerprint);
  options.backend_compiler_revision = std::move(backend_compiler_revision);
  options.driver_identity = std::move(driver_identity);
  return options;
}

compiler::PlannedOperation
make_metadata_planned_operation(const std::shared_ptr<const ov::Node> &node,
                                compiler::TensorLayoutPlan layout,
                                bool requires_runtime_shape_args = false) {
  compiler::PlannedOperation op;
  op.source_node = node;
  op.node_name = node ? node->get_friendly_name() : "metadata";
  op.type_name = node ? node->get_type_name() : "Unknown";
  op.kernel_unit = compiler::KernelUnit::describe(
      LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
      "opencl", "metadata", requires_runtime_shape_args);
  op.layout = layout;
  op.profitability_score = 1.0;
  op.input_element_types = {"f32"};
  op.input_shapes = {"{1,2,3}"};
  op.output_element_types = {"f32"};
  op.output_shapes = {"{1,2,3}"};
  return op;
}

RuntimeStageExecutableDescriptor
make_runtime_descriptor_for_layout(const std::shared_ptr<const ov::Node> &node,
                                   compiler::TensorLayoutPlan layout) {
  compiler::LoweringPlan plan;
  plan.target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  plan.operations.push_back(make_metadata_planned_operation(node, layout));
  const auto manifest = compiler::ManifestBuilder{}.build(plan);
  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  const auto runtime_descriptor =
      compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
  EXPECT_EQ(runtime_descriptor.stages.size(), 1u);
  return runtime_descriptor.stages.front();
}

TEST_F(GfxBackendArchitectureContractTest,
       KnownTargetsUseConcreteOopIdentityWithoutInverseBuckets) {
  std::unordered_set<std::string> fingerprints;
  bool saw_metal_apple = false;
  bool saw_opencl_generic = false;
  bool saw_opencl_adreno = false;
  bool saw_opencl_v3d = false;

  for (const auto &target_contract : backend_catalog.known_target_contracts()) {
    EXPECT_TRUE(target_contract.has_concrete_oop_identity())
        << target_contract.target().debug_string();
    EXPECT_TRUE(target_contract.has_profiled_cache_identity())
        << target_contract.target().fingerprint();
    EXPECT_TRUE(target_contract.avoids_inverse_apple_bucket())
        << target_contract.target().debug_string();
    EXPECT_TRUE(fingerprints.insert(target_contract.target().fingerprint()).second)
        << target_contract.target().debug_string();

    saw_metal_apple |=
        target_contract.target().device_profile() == "metal_apple";
    saw_opencl_generic |=
        target_contract.target().device_profile() == "opencl_generic";
    saw_opencl_adreno |=
        target_contract.target().device_profile() == "opencl_adreno";
    saw_opencl_v3d |=
        target_contract.target().device_profile() == "opencl_broadcom_v3d";
  }
  EXPECT_TRUE(saw_metal_apple);
  EXPECT_TRUE(saw_opencl_generic);
  EXPECT_TRUE(saw_opencl_adreno);
  EXPECT_TRUE(saw_opencl_v3d);
}

TEST_F(GfxBackendArchitectureContractTest,
       AppleTransposedSdpaVendorContractOwnsShapeAndTypeContract) {
  GfxAppleMpsVendorPrimitiveContract contract;
  EXPECT_TRUE(gfx_apple_make_mps_transposed_sdpa_contract(
      "attention", ov::element::f32, {1, 2, 3, 4}, {1, 2, 3, 5}, {1, 2, 6, 5},
      {1, 2, 6, 4}, 0.5f, contract));
  EXPECT_TRUE(contract.valid);
  EXPECT_EQ(contract.descriptor.kind, GfxAppleMpsVendorPrimitiveKind::Sdpa);
  EXPECT_EQ(contract.descriptor.sdpa.layout, GfxMpsrtSdpaLayoutTransposedBHDN);
  EXPECT_FLOAT_EQ(contract.descriptor.sdpa.scale, 0.5f);
  ASSERT_TRUE(contract.external_buffer_abi.valid);
  EXPECT_EQ(contract.external_buffer_abi.buffer_count, 4u);
  EXPECT_EQ(contract.external_buffer_abi.output_buffer_count, 1u);

  EXPECT_FALSE(gfx_apple_make_mps_transposed_sdpa_contract(
      "attention", ov::element::i32, {1, 2, 3, 4}, {1, 2, 3, 5}, {1, 2, 6, 5},
      {1, 2, 6, 4}, 0.5f, contract));
  EXPECT_FALSE(gfx_apple_make_mps_transposed_sdpa_contract(
      "attention", ov::element::f32, {1, 2, 3, 4}, {1, 2, 3, 5}, {1, 2, 6, 5},
      {1, 2, 7, 4}, 0.5f, contract));
  EXPECT_FALSE(gfx_apple_make_mps_transposed_sdpa_contract(
      "attention", ov::element::f32, {1, 2, 3, 4}, {1, 2, 3}, {1, 2, 6, 5},
      {1, 2, 6, 4}, 0.5f, contract));
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerRegistryContainsAllProductionBackendModules) {
  const auto &registry = compiler::BackendRegistry::default_registry();
  const auto metal_target =
      compiler::BackendTarget::from_backend(GpuBackend::Metal);
  const auto opencl_target =
      compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const auto metal = registry.resolve(metal_target);
  const auto opencl = registry.resolve(opencl_target);
  const auto targets = registry.available_targets();
  const auto expected_target_count =
      static_cast<size_t>(kGfxBackendMetalAvailable) +
      (kGfxBackendOpenCLAvailable ? size_t{3} : size_t{0});

  EXPECT_EQ(static_cast<bool>(metal), kGfxBackendMetalAvailable);
  EXPECT_EQ(static_cast<bool>(opencl), kGfxBackendOpenCLAvailable);
  EXPECT_EQ(targets.size(), expected_target_count);

  if (metal) {
    EXPECT_TRUE(metal->capabilities().stage_placement());
    EXPECT_EQ(metal->target().fingerprint(), metal_target.fingerprint());
  }
  if (opencl) {
    EXPECT_TRUE(opencl->capabilities().stage_placement());
    EXPECT_EQ(opencl->target().fingerprint(), opencl_target.fingerprint());
    EXPECT_EQ(opencl->target().device_profile(), "opencl_generic");

    const auto adreno_target =
        compiler::BackendTarget::from_backend_device_family(
            GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno);
    const auto v3d_target = compiler::BackendTarget::from_backend_device_family(
        GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D);
    const auto adreno = registry.resolve(adreno_target);
    const auto v3d = registry.resolve(v3d_target);
    ASSERT_TRUE(adreno);
    ASSERT_TRUE(v3d);
    EXPECT_EQ(adreno->target().fingerprint(), adreno_target.fingerprint());
    EXPECT_EQ(v3d->target().fingerprint(), v3d_target.fingerprint());
    EXPECT_EQ(v3d->target().runtime_api(), kBackendOpenCL);
    EXPECT_EQ(v3d->target().driver_id(), "opencl-clvk");
    EXPECT_NE(opencl->target().fingerprint(), v3d->target().fingerprint());
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       BackendCapabilitiesDoNotAdvertiseCompiledModelExportImportBeforeEnvelopeRoundTrip) {
  const auto module_contracts = backend_catalog.compiled_module_contracts();
  ASSERT_FALSE(module_contracts.empty());
  for (const auto &module_contract : module_contracts) {
    const auto &capabilities = module_contract.module().capabilities();
    EXPECT_FALSE(capabilities.artifact_formats()
                     .supports_compiled_model_export_import)
        << module_contract.target().debug_string();
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       BackendModulesOwnVendorAttentionArtifactContracts) {
  compiler::PipelineVendorAttentionPlan plan;
  plan.name = "test_vendor_attention";
  plan.element_type = ov::element::f32;
  plan.query_shape = {1, 2, 3, 4};
  plan.key_shape = {1, 2, 3, 5};
  plan.value_shape = {1, 2, 6, 5};
  plan.output_shape = {1, 2, 6, 4};
  plan.scale = 0.5f;

  const auto &registry = compiler::BackendRegistry::default_registry();
  if (const auto metal =
          registry.resolve(compiler::BackendTarget::from_backend(GpuBackend::Metal))) {
    auto artifact = metal->materialize_vendor_attention_artifact(0x1234u, plan);
    ASSERT_TRUE(artifact.valid());
    EXPECT_EQ(artifact.descriptor.stage_record_key, 0x1234u);
    EXPECT_EQ(artifact.descriptor.kernel.kernel_id,
              "metal/vendor/mpsgraph_sdpa");
    EXPECT_EQ(artifact.descriptor.kernel.op_family, "VendorAttention");
    EXPECT_EQ(artifact.descriptor.kernel.backend_domain, "metal");
    EXPECT_EQ(artifact.descriptor.kernel.origin,
              compiler::KernelArtifactOrigin::VendorPrimitive);
    EXPECT_EQ(artifact.descriptor.payload_kind,
              compiler::KernelArtifactPayloadKind::VendorDescriptor);
    EXPECT_EQ(artifact.descriptor.entry_point, "mps_sdpa");
    EXPECT_EQ(artifact.descriptor.compile_options_key,
              "metal_vendor_descriptor");
    EXPECT_FALSE(artifact.descriptor.artifact_key.empty());
  }

  if (const auto opencl =
          registry.resolve(compiler::BackendTarget::from_backend(GpuBackend::OpenCL))) {
    EXPECT_FALSE(
        opencl->materialize_vendor_attention_artifact(0x1234u, plan).valid());
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       SharedBackendValueObjectsDoNotDefaultToMetal) {
  EXPECT_EQ(static_cast<int>(GpuBackend::Metal), 0);
  EXPECT_EQ(static_cast<int>(GpuBackend::OpenCL), 1);
  EXPECT_NE(static_cast<int>(GpuBackend::Unknown),
            static_cast<int>(GpuBackend::Metal));

  compiler::BackendTarget target;
  EXPECT_EQ(target.backend(), GpuBackend::Unknown);
  EXPECT_TRUE(target.backend_id().empty());

  compiler::GfxCompileRequest compile_request;
  EXPECT_EQ(compile_request.target.backend(), GpuBackend::Unknown);

  compiler::PipelineStageBuildRequest stage_build_request;
  EXPECT_EQ(stage_build_request.target.backend(), GpuBackend::Unknown);

  compiler::StageCompilerPolicy stage_compiler_policy;
  EXPECT_EQ(stage_compiler_policy.target.backend(), GpuBackend::Unknown);
  EXPECT_EQ(stage_compiler_policy.backend, GpuBackend::Unknown);

  compiler::LoweringPlan lowering_plan;
  EXPECT_EQ(lowering_plan.target.backend(), GpuBackend::Unknown);

  GpuBuffer buffer;
  EXPECT_EQ(buffer.backend, GpuBackend::Unknown);
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerOwnsBackendNeutralTensorLayoutClassification) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 2, 3});
  auto identity_perm =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 2});
  auto identity_transpose =
      std::make_shared<ov::op::v1::Transpose>(input, identity_perm);
  const auto identity_plan = compiler::select_tensor_layout_plan(
      identity_transpose->get_type_name(), identity_transpose);
  EXPECT_TRUE(identity_plan.view_only);
  EXPECT_EQ(identity_plan.kind, GfxTensorLayoutKind::ViewOnly);

  auto materialized_perm =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
  auto materialized_transpose =
      std::make_shared<ov::op::v1::Transpose>(input, materialized_perm);
  const auto materialized_plan = compiler::select_tensor_layout_plan(
      materialized_transpose->get_type_name(), materialized_transpose);
  EXPECT_FALSE(materialized_plan.view_only);
  EXPECT_EQ(materialized_plan.kind, GfxTensorLayoutKind::Materialized);

  const auto reshape_plan =
      compiler::select_tensor_layout_plan("Reshape", nullptr);
  EXPECT_TRUE(reshape_plan.view_only);
  EXPECT_EQ(reshape_plan.kind, GfxTensorLayoutKind::ViewOnly);

  const auto read_value_plan =
      compiler::select_tensor_layout_plan("ReadValue", nullptr);
  EXPECT_FALSE(read_value_plan.view_only);
  EXPECT_EQ(read_value_plan.kind, GfxTensorLayoutKind::Materialized);

  const auto view_descriptor =
      make_runtime_descriptor_for_layout(identity_transpose, identity_plan);
  EXPECT_EQ(view_descriptor.layout_contract, "view_only");
  EXPECT_TRUE(view_descriptor.tensor_view_only);

  const auto materialized_descriptor = make_runtime_descriptor_for_layout(
      materialized_transpose, materialized_plan);
  EXPECT_EQ(materialized_descriptor.layout_contract, "materialized");
  EXPECT_FALSE(materialized_descriptor.tensor_view_only);
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerOwnsMemoryPlanInManifestExecutableAndCacheIdentity) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 2, 3});
  const auto layout = compiler::select_tensor_layout_plan("Relu", input);
  compiler::LoweringPlan lowering_plan;
  lowering_plan.target =
      compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  lowering_plan.operations.push_back(
      make_metadata_planned_operation(input, layout));

  const auto manifest = compiler::ManifestBuilder{}.build(lowering_plan);
  ASSERT_TRUE(manifest.verify().valid());
  ASSERT_TRUE(manifest.memory_plan.valid());
  ASSERT_EQ(manifest.stages.size(), 1u);
  const auto &stage = manifest.stages.front();
  ASSERT_FALSE(stage.inputs.empty());
  ASSERT_FALSE(stage.outputs.empty());
  EXPECT_TRUE(
      manifest.memory_plan.has_region(stage.inputs.front().memory_region_id));
  EXPECT_TRUE(
      manifest.memory_plan.has_region(stage.outputs.front().memory_region_id));
  EXPECT_TRUE(manifest.memory_plan.has_alias_group(stage.memory.alias_group));
  EXPECT_FALSE(manifest.memory_plan.hidden_host_copies_allowed);

  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.verify().valid());
  EXPECT_EQ(compiler::make_memory_plan_fingerprint(executable.memory_plan),
            compiler::make_memory_plan_fingerprint(manifest.memory_plan));

  auto stale_manifest = manifest;
  ASSERT_FALSE(stale_manifest.memory_plan.regions.empty());
  stale_manifest.memory_plan.regions.front().layout = "stale_layout";
  EXPECT_NE(compiler::make_manifest_cache_hash(stale_manifest),
            compiler::make_manifest_cache_hash(manifest));
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeExecutableDescriptorCarriesCompilerOwnedMemoryPlan) {
  for (const auto *backend_domain : {"opencl", "metal"}) {
    SCOPED_TRACE(backend_domain);
    const auto manifest = make_single_payload_route_manifest(
        LoweringRouteKind::Metadata, backend_domain, "metadata", "metadata");
    ASSERT_TRUE(manifest.valid());

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    ASSERT_TRUE(executable.valid());
    ASSERT_FALSE(executable.memory_plan.regions.empty());
    ASSERT_FALSE(executable.memory_plan.alias_groups.empty());
    ASSERT_FALSE(executable.memory_plan.transient_arenas.empty());

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    const auto verification = compiler::verify_runtime_executable_descriptor(
        runtime_descriptor, executable);
    ASSERT_TRUE(verification.valid())
        << (verification.diagnostics.empty()
                ? std::string{}
                : verification.diagnostics.front());

    EXPECT_EQ(runtime_descriptor.memory_plan.schema_version,
              executable.memory_plan.schema_version);
    EXPECT_EQ(runtime_descriptor.memory_plan.fingerprint,
              compiler::make_memory_plan_fingerprint(executable.memory_plan));
    EXPECT_EQ(runtime_descriptor.memory_plan.regions.size(),
              executable.memory_plan.regions.size());
    EXPECT_EQ(runtime_descriptor.memory_plan.alias_groups.size(),
              executable.memory_plan.alias_groups.size());
    EXPECT_EQ(runtime_descriptor.memory_plan.transient_arenas.size(),
              executable.memory_plan.transient_arenas.size());
    EXPECT_FALSE(runtime_descriptor.memory_plan.hidden_host_copies_allowed);
    EXPECT_TRUE(runtime_descriptor.memory_plan.has_region(
        executable.memory_plan.regions.front().region_id));
    EXPECT_TRUE(runtime_descriptor.memory_plan.has_alias_group(
        executable.memory_plan.alias_groups.front().group_id));
    EXPECT_EQ(runtime_descriptor.memory_plan.regions.front().layout,
              executable.memory_plan.regions.front().layout);
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &runtime_stage = runtime_descriptor.stages.front();
    ASSERT_EQ(runtime_stage.input_bindings.size(), 1u);
    ASSERT_EQ(runtime_stage.output_bindings.size(), 1u);
    EXPECT_EQ(
        runtime_stage.input_bindings.front().memory_region_id,
        executable.manifest.stages.front().inputs.front().memory_region_id);
    EXPECT_EQ(
        runtime_stage.output_bindings.front().memory_region_id,
        executable.manifest.stages.front().outputs.front().memory_region_id);
    EXPECT_EQ(runtime_stage.input_bindings.front().alias_group,
              executable.memory_plan.regions.front().alias_group);

    std::vector<GpuTensor *> input_slots(runtime_stage.input_bindings.size(),
                                         nullptr);
    std::vector<GpuTensor *> output_slots(runtime_stage.output_bindings.size(),
                                          nullptr);
    const auto binding_table = ResourceBindingTable::for_stage(
        input_slots, output_slots, runtime_stage);
    EXPECT_TRUE(binding_table.compatible_with(runtime_stage));
    EXPECT_EQ(binding_table.input_region_ids().front(),
              runtime_stage.input_bindings.front().memory_region_id);
    EXPECT_EQ(binding_table.output_region_ids().front(),
              runtime_stage.output_bindings.front().memory_region_id);

    output_slots.clear();
    const auto incomplete_binding_table = ResourceBindingTable::for_stage(
        input_slots, output_slots, runtime_stage);
    EXPECT_FALSE(incomplete_binding_table.compatible_with(runtime_stage));

    auto stale_region_descriptor = runtime_descriptor;
    stale_region_descriptor.memory_plan.regions.front().layout =
        "stale_runtime_layout";
    const auto stale_region_verification =
        compiler::verify_runtime_executable_descriptor(stale_region_descriptor,
                                                       executable);
    EXPECT_FALSE(stale_region_verification.valid());
    EXPECT_TRUE(has_diagnostic_containing(stale_region_verification.diagnostics,
                                          "memory region drift"));

    auto stale_fingerprint_descriptor = runtime_descriptor;
    stale_fingerprint_descriptor.memory_plan.fingerprint =
        "stale-runtime-memory-plan";
    const auto stale_fingerprint_verification =
        compiler::verify_runtime_executable_descriptor(
            stale_fingerprint_descriptor, executable);
    EXPECT_FALSE(stale_fingerprint_verification.valid());
    EXPECT_TRUE(
        has_diagnostic_containing(stale_fingerprint_verification.diagnostics,
                                  "memory plan fingerprint drift"));

    auto stale_binding_descriptor = runtime_descriptor;
    stale_binding_descriptor.stages.front()
        .input_bindings.front()
        .memory_region_id = "stale-runtime-input-region";
    const auto stale_binding_verification =
        compiler::verify_runtime_executable_descriptor(stale_binding_descriptor,
                                                       executable);
    EXPECT_FALSE(stale_binding_verification.valid());
    EXPECT_TRUE(has_diagnostic_containing(
        stale_binding_verification.diagnostics, "input binding drift"));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStageMaterializerPreservesCompilerOwnedMaterializedBindings) {
  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto relu = std::make_shared<ov::op::v0::Relu>(parameter);
  const std::vector<std::shared_ptr<ov::Node>> ordered_ops{relu};

  RuntimeExecutableDescriptor runtime_descriptor;
  runtime_descriptor.target_fingerprint = "metal:unit";
  runtime_descriptor.stages.push_back(
      make_materializer_base_descriptor(relu));

  auto vendor_descriptor = runtime_descriptor.stages.front();
  vendor_descriptor.input_bindings = {make_runtime_binding(
      "vendor.input0", "compiler_vendor_input_region", "TensorInput")};
  vendor_descriptor.output_bindings = {make_runtime_binding(
      "vendor.output0", "compiler_vendor_output_region", "TensorOutput")};
  vendor_descriptor.abi_arg_count = 1;
  vendor_descriptor.abi_output_arg_count = 1;

  const auto plan =
      make_vendor_materialization_plan(parameter, relu, vendor_descriptor);
  UnitBackendStageFactory stage_factory;
  PipelineStageMaterializer materializer(stage_factory, ordered_ops,
                                        runtime_descriptor, {});

  const auto materialized = materializer.create_materialized_descriptor(plan);
  ASSERT_TRUE(materialized);
  ASSERT_EQ(materialized->input_bindings.size(), 1u);
  ASSERT_EQ(materialized->output_bindings.size(), 1u);
  EXPECT_EQ(materialized->input_bindings.front().memory_region_id,
            "compiler_vendor_input_region");
  EXPECT_EQ(materialized->output_bindings.front().memory_region_id,
            "compiler_vendor_output_region");
}

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStageMaterializerRejectsIncompleteMaterializedBindings) {
  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto relu = std::make_shared<ov::op::v0::Relu>(parameter);
  const std::vector<std::shared_ptr<ov::Node>> ordered_ops{relu};

  RuntimeExecutableDescriptor runtime_descriptor;
  runtime_descriptor.target_fingerprint = "metal:unit";
  runtime_descriptor.stages.push_back(
      make_materializer_base_descriptor(relu));

  auto incomplete_vendor_descriptor = runtime_descriptor.stages.front();
  incomplete_vendor_descriptor.input_bindings.clear();
  incomplete_vendor_descriptor.output_bindings.clear();
  incomplete_vendor_descriptor.abi_arg_count = 0;
  incomplete_vendor_descriptor.abi_output_arg_count = 0;

  const auto plan = make_vendor_materialization_plan(
      parameter, relu, incomplete_vendor_descriptor);
  UnitBackendStageFactory stage_factory;
  PipelineStageMaterializer materializer(stage_factory, ordered_ops,
                                        runtime_descriptor, {});

  EXPECT_THROW((void)materializer.create_materialized_descriptor(plan),
               ov::Exception);
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeExecutableDescriptorRequiresFrozenCompilerStagePlan) {
  const auto manifest = make_single_payload_route_manifest(
      LoweringRouteKind::Metadata, "opencl", "metadata", "metadata");
  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.valid());

  const auto descriptor_without_stage_plan =
      compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
  ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
      descriptor_without_stage_plan, executable));

  const auto stage_plan_verification =
      compiler::verify_runtime_executable_stage_plan(
          descriptor_without_stage_plan);
  EXPECT_FALSE(stage_plan_verification.valid());
  EXPECT_TRUE(has_diagnostic_containing(stage_plan_verification.diagnostics,
                                        "missing compiler-owned stage plan"));

  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto relu = std::make_shared<ov::op::v0::Relu>(parameter);
  auto unfrozen_stage_plan = std::make_shared<PipelineStageRuntimePlan>();
  unfrozen_stage_plan->ordered_ops = {relu};
  PipelineStageMaterializationPlan materialized_stage;
  materialized_stage.kind = PipelineStageMaterializationKind::SingleStage;
  materialized_stage.io_plan.node = relu;
  materialized_stage.io_plan.runtime_stage_index = 0;
  unfrozen_stage_plan->stage_plans.push_back(std::move(materialized_stage));

  auto descriptor_with_unfrozen_stage_plan = descriptor_without_stage_plan;
  descriptor_with_unfrozen_stage_plan.stage_plan = unfrozen_stage_plan;
  const auto unfrozen_verification =
      compiler::verify_runtime_executable_stage_plan(
          descriptor_with_unfrozen_stage_plan);
  EXPECT_FALSE(unfrozen_verification.valid());
  EXPECT_TRUE(has_diagnostic_containing(unfrozen_verification.diagnostics,
                                        "materialized descriptor missing"));
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeSliceShapeArgsComeFromKernelUnitManifestAndDescriptor) {
  struct Case {
    GpuBackend backend;
    bool requires_runtime_shape_args;
  };

  for (const auto test_case :
       {Case{GpuBackend::OpenCL, true}, Case{GpuBackend::Metal, false}}) {
    SCOPED_TRACE(backend_to_string(test_case.backend));
    compiler::LoweringPlan plan;
    plan.target = compiler::BackendTarget::from_backend(test_case.backend);
    compiler::PlannedOperation op;
    op.node_name = "Slice";
    op.type_name = "Slice";
    op.kernel_unit = compiler::KernelUnit::describe(
        LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
        plan.target.backend_id(), "Slice",
        test_case.requires_runtime_shape_args);
    op.layout = compiler::TensorLayoutPlan{};
    op.profitability_score = 1.0;
    op.input_element_types = {"f32"};
    op.input_shapes = {"{1,2,3}"};
    op.output_element_types = {"f32"};
    op.output_shapes = {"{1,2,3}"};
    plan.operations.push_back(std::move(op));

    const auto manifest = compiler::ManifestBuilder{}.build(plan);
    ASSERT_TRUE(manifest.valid());
    ASSERT_EQ(manifest.stages.size(), 1u);
    EXPECT_EQ(manifest.stages.front().requires_runtime_shape_args,
              test_case.requires_runtime_shape_args);

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    ASSERT_TRUE(executable.valid());
    ASSERT_EQ(executable.artifact_descriptors.size(), 1u);
    EXPECT_EQ(executable.artifact_descriptors.front()
                  .kernel.requires_runtime_shape_args,
              test_case.requires_runtime_shape_args);

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().requires_runtime_shape_args,
              test_case.requires_runtime_shape_args);

    auto stale_descriptor = runtime_descriptor;
    stale_descriptor.stages.front().requires_runtime_shape_args =
        !test_case.requires_runtime_shape_args;
    const auto stale_result = compiler::verify_runtime_executable_descriptor(
        stale_descriptor, executable);
    EXPECT_FALSE(stale_result.valid());
    EXPECT_TRUE(
        has_diagnostic_containing(stale_result.diagnostics, "artifact drift"));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeShapeRuleComesFromCompilerManifestAndDescriptor) {
  struct Case {
    const char *op_type;
    const char *expected_rule;
  };

  for (const auto test_case :
       {Case{"Concat", "concat"}, Case{"Broadcast", "broadcast"},
        Case{"Select", "select"}, Case{"ShapeOf", "shape_of"},
        Case{"Slice", "slice"}, Case{"StridedSlice", "slice"},
        Case{"Range", "range"}, Case{"Tile", "tile"},
        Case{"Relu", "static_or_descriptor"}}) {
    SCOPED_TRACE(test_case.op_type);
    compiler::LoweringPlan plan;
    plan.target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
    compiler::PlannedOperation op;
    op.node_name = test_case.op_type;
    op.type_name = test_case.op_type;
    op.kernel_unit = compiler::KernelUnit::describe(
        LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
        plan.target.backend_id(), test_case.op_type);
    op.layout = compiler::TensorLayoutPlan{};
    op.profitability_score = 1.0;
    op.input_element_types = {"f32"};
    op.input_shapes = {"{1,2,3}"};
    op.output_element_types = {"f32"};
    op.output_shapes = {"{1,2,3}"};
    plan.operations.push_back(std::move(op));

    const auto manifest = compiler::ManifestBuilder{}.build(plan);
    ASSERT_TRUE(manifest.valid());
    ASSERT_EQ(manifest.stages.size(), 1u);
    EXPECT_EQ(manifest.stages.front().runtime_shape.rule,
              test_case.expected_rule);

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    ASSERT_TRUE(executable.valid());
    ASSERT_EQ(executable.artifact_descriptors.size(), 1u);
    EXPECT_EQ(executable.artifact_descriptors.front().kernel.runtime_shape_rule,
              test_case.expected_rule);

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().runtime_shape_rule,
              test_case.expected_rule);

    auto stale_descriptor = runtime_descriptor;
    stale_descriptor.stages.front().runtime_shape_rule = "stale_runtime_rule";
    const auto stale_result = compiler::verify_runtime_executable_descriptor(
        stale_descriptor, executable);
    EXPECT_FALSE(stale_result.valid());
    EXPECT_TRUE(
        has_diagnostic_containing(stale_result.diagnostics, "artifact drift"));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeSubmissionContractComesFromCompilerManifestAndDescriptor) {
  struct Case {
    const char *op_type;
    bool expected_dependency_boundary;
  };

  for (const auto test_case :
       {Case{"Concat", true}, Case{"Softmax", true}, Case{"Relu", false}}) {
    SCOPED_TRACE(test_case.op_type);
    compiler::LoweringPlan plan;
    plan.target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
    compiler::PlannedOperation op;
    op.node_name = test_case.op_type;
    op.type_name = test_case.op_type;
    op.kernel_unit = compiler::KernelUnit::describe(
        LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
        plan.target.backend_id(), test_case.op_type);
    op.layout = compiler::TensorLayoutPlan{};
    op.profitability_score = 1.0;
    op.input_element_types = {"f32"};
    op.input_shapes = {"{1,2,3}"};
    op.output_element_types = {"f32"};
    op.output_shapes = {"{1,2,3}"};
    plan.operations.push_back(std::move(op));

    const auto manifest = compiler::ManifestBuilder{}.build(plan);
    ASSERT_TRUE(manifest.valid());
    ASSERT_EQ(manifest.stages.size(), 1u);
    EXPECT_EQ(manifest.stages.front().submission.stage_weight, 1u);
    EXPECT_EQ(manifest.stages.front().submission.dependency_extension_boundary,
              test_case.expected_dependency_boundary);

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    ASSERT_TRUE(executable.valid());

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().submission_stage_weight,
              manifest.stages.front().submission.stage_weight);
    EXPECT_EQ(runtime_descriptor.stages.front().submission_macs_estimate,
              manifest.stages.front().submission.macs_estimate);
    EXPECT_EQ(runtime_descriptor.stages.front().submission_dependency_boundary,
              test_case.expected_dependency_boundary);

    auto stale_descriptor = runtime_descriptor;
    stale_descriptor.stages.front().submission_dependency_boundary =
        !test_case.expected_dependency_boundary;
    const auto stale_result = compiler::verify_runtime_executable_descriptor(
        stale_descriptor, executable);
    EXPECT_FALSE(stale_result.valid());
    EXPECT_TRUE(
        has_diagnostic_containing(stale_result.diagnostics, "artifact drift"));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       CacheEnvelopeContainsDocsRequiredCompilerOwnedIdentity) {
  const auto manifest = make_single_payload_route_manifest(
      LoweringRouteKind::Metadata, "opencl", "metadata", "metadata");
  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.verify().valid());

  const auto model = make_relu_model();
  const auto envelope = compiler::CacheEnvelopeBuilder{}.build(
      executable, make_test_cache_options(*model));

  const auto verification = envelope.verify(executable);
  EXPECT_TRUE(verification.valid())
      << (verification.diagnostics.empty() ? std::string{}
                                           : verification.diagnostics.front());
  EXPECT_FALSE(envelope.key.model_fingerprint.empty());
  EXPECT_EQ(envelope.key.manifest_hash,
            compiler::make_manifest_cache_hash(manifest));
  EXPECT_EQ(envelope.key.target_fingerprint, manifest.target_fingerprint);
  EXPECT_EQ(envelope.key.compile_options_hash,
            compiler::make_executable_compile_options_hash(executable));
  EXPECT_FALSE(envelope.key.backend_capabilities_fingerprint.empty());
  EXPECT_FALSE(envelope.key.compiler_revision.empty());
  EXPECT_FALSE(envelope.key.backend_compiler_revision.empty());
  EXPECT_FALSE(envelope.key.driver_identity.empty());
  EXPECT_FALSE(envelope.key.kernel_unit_versions.empty());
  EXPECT_FALSE(envelope.key.stable_key.empty());
  EXPECT_EQ(envelope.manifest.target_fingerprint, manifest.target_fingerprint);
  EXPECT_EQ(envelope.artifact_descriptors.size(),
            executable.artifact_descriptors.size());
}

TEST_F(GfxBackendArchitectureContractTest,
       BackendDeviceProfilesParticipateInCacheAndCapabilityIdentity) {
  const auto generic_target =
      compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const auto adreno_target =
      compiler::BackendTarget::from_backend_device_family(
          GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno);
  const auto v3d_target = compiler::BackendTarget::from_backend_device_family(
      GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D);

  EXPECT_EQ(generic_target.runtime_api(), kBackendOpenCL);
  EXPECT_EQ(adreno_target.runtime_api(), kBackendOpenCL);
  EXPECT_EQ(v3d_target.runtime_api(), kBackendOpenCL);
  EXPECT_NE(generic_target.fingerprint(), adreno_target.fingerprint());
  EXPECT_NE(generic_target.fingerprint(), v3d_target.fingerprint());
  EXPECT_NE(adreno_target.fingerprint(), v3d_target.fingerprint());
  EXPECT_EQ(adreno_target.device_profile(), "opencl_adreno");
  EXPECT_EQ(v3d_target.device_profile(), "opencl_broadcom_v3d");
  EXPECT_TRUE(target_has_feature(adreno_target, "adreno"));
  EXPECT_TRUE(target_has_feature(v3d_target, "clvk"));
  EXPECT_TRUE(target_has_feature(v3d_target, "clspv"));

  const auto generic_module =
      compiler::make_opencl_backend_module(generic_target);
  const auto adreno_module = compiler::make_opencl_backend_module(adreno_target);
  const auto v3d_module = compiler::make_opencl_backend_module(v3d_target);
  ASSERT_TRUE(generic_module);
  ASSERT_TRUE(adreno_module);
  ASSERT_TRUE(v3d_module);

  EXPECT_EQ(generic_module->target().fingerprint(),
            generic_target.fingerprint());
  EXPECT_EQ(adreno_module->target().fingerprint(),
            adreno_target.fingerprint());
  EXPECT_EQ(v3d_module->target().fingerprint(), v3d_target.fingerprint());

  const auto &generic_profile =
      generic_module->capabilities().execution().custom_kernel_dispatch_profile;
  const auto &adreno_profile =
      adreno_module->capabilities().execution().custom_kernel_dispatch_profile;
  const auto &v3d_profile =
      v3d_module->capabilities().execution().custom_kernel_dispatch_profile;

  EXPECT_EQ(generic_profile.profile_key, "opencl:generic");
  EXPECT_EQ(adreno_profile.profile_key, "opencl:adreno");
  EXPECT_EQ(v3d_profile.profile_key, "opencl:broadcom_v3d");
  EXPECT_TRUE(adreno_profile.supports_conv_output_channel_blocking);
  EXPECT_TRUE(adreno_profile.supports_conv_channel_block_spatial_tiling);
  EXPECT_TRUE(v3d_profile.enable_skinny_matmul_tiles);
  EXPECT_TRUE(v3d_profile.chunk_dispatch.retune_threads_to_workload);
  EXPECT_LT(v3d_profile.max_total_threads_per_group,
            adreno_profile.max_total_threads_per_group);

  EXPECT_NE(compiler::make_backend_capabilities_fingerprint(
                generic_module->capabilities()),
            compiler::make_backend_capabilities_fingerprint(
                adreno_module->capabilities()));
  EXPECT_NE(compiler::make_backend_capabilities_fingerprint(
                generic_module->capabilities()),
            compiler::make_backend_capabilities_fingerprint(
                v3d_module->capabilities()));
  EXPECT_NE(compiler::make_backend_capabilities_fingerprint(
                adreno_module->capabilities()),
            compiler::make_backend_capabilities_fingerprint(
                v3d_module->capabilities()));
}

TEST_F(GfxBackendArchitectureContractTest,
       OpenClParallelismProfileRejectsAppleDeviceFamily) {
  EXPECT_THROW(make_opencl_parallelism_profile(GpuDeviceFamily::Apple),
               ov::Exception);
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeStagePlanningPreservesConcreteBackendTargetProfile) {
  const auto target = compiler::BackendTarget::from_backend_device_family(
      GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D);
  const compiler::BackendRegistry registry(
      {compiler::make_opencl_backend_module(target)});
  const auto module = registry.resolve(target);
  ASSERT_TRUE(module);
  EXPECT_EQ(module->capabilities()
                .execution()
                .custom_kernel_dispatch_profile.profile_key,
            "opencl:broadcom_v3d");

  const compiler::GfxCompilerService compiler_service(registry);
  const auto compile_result =
      compiler_service.compile({make_static_range_model(), target});
  ASSERT_TRUE(compile_result.supported())
      << compile_result.unsupported_message();
  EXPECT_EQ(compile_result.target.fingerprint(), target.fingerprint());
  EXPECT_EQ(compile_result.executable.target_fingerprint, target.fingerprint());

  ASSERT_TRUE(compile_result.runtime_descriptor);
  EXPECT_EQ(compile_result.runtime_descriptor->target_fingerprint,
            target.fingerprint());
  ASSERT_TRUE(compile_result.runtime_descriptor->stage_plan);
  EXPECT_EQ(compile_result.runtime_descriptor->stage_plan->ordered_ops.size(),
            compile_result.runtime_descriptor->stages.size());
  EXPECT_EQ(compile_result.runtime_descriptor->stage_plan
                ->runtime_options.custom_kernel_dispatch_profile
                .profile_key,
            "opencl:broadcom_v3d");
  EXPECT_EQ(compile_result.runtime_descriptor->stage_plan
                ->runtime_options.custom_kernel_dispatch_profile
                .max_total_threads_per_group,
            64u);
  EXPECT_TRUE(compile_result.runtime_descriptor->stage_plan
                  ->runtime_options.custom_kernel_dispatch_profile
                  .chunk_dispatch.retune_threads_to_workload);
}

TEST_F(GfxBackendArchitectureContractTest,
       BackendRegistryRequiresExactConcreteBackendTargetProfiles) {
  const auto generic_target =
      compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  compiler::StaticBackendModuleConfig config;
  config.target = generic_target;
  config.kernel_registry = compiler::make_common_kernel_registry(generic_target);
  const compiler::BackendRegistry generic_only_registry(
      {compiler::make_static_backend_module(std::move(config))});
  const compiler::GfxCompilerService compiler_service(generic_only_registry);

  const auto v3d_target = compiler::BackendTarget::from_backend_device_family(
      GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D);
  ASSERT_TRUE(generic_only_registry.resolve(generic_target));
  EXPECT_FALSE(generic_only_registry.resolve(v3d_target));
  EXPECT_THROW((void)compiler_service.compile(
                   {models.passthrough(ov::PartialShape{1}), v3d_target}),
               ov::Exception);
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeStagePlanningRequiresExplicitBackendRegistry) {
  const auto target = compiler::BackendTarget::from_backend_device_family(
      GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D);
  const compiler::BackendRegistry registry(
      {compiler::make_opencl_backend_module(target)});
  const compiler::GfxCompilerService compiler_service(registry);

  const auto compile_result =
      compiler_service.compile({make_static_range_model(), target});
  ASSERT_TRUE(compile_result.supported())
      << compile_result.unsupported_message();
  ASSERT_TRUE(compile_result.runtime_descriptor);

  compiler::PipelineStageBuildRequest stage_request;
  stage_request.runtime_model = compile_result.transformed_model;
  stage_request.runtime_descriptor = compile_result.runtime_descriptor.get();
  stage_request.target = target;

  EXPECT_THROW((void)compiler::build_pipeline_stage_runtime_plan(stage_request),
               ov::Exception);

  stage_request.backend_registry = &registry;
  const auto stage_plan = compiler::build_pipeline_stage_runtime_plan(stage_request);
  ASSERT_TRUE(stage_plan);
  EXPECT_EQ(stage_plan->runtime_options.custom_kernel_dispatch_profile.profile_key,
            "opencl:broadcom_v3d");
}

TEST_F(GfxBackendArchitectureContractTest,
       CacheEnvelopeRejectsStaleManifestDriverAndKernelIdentity) {
  const auto manifest = make_single_payload_route_manifest(
      LoweringRouteKind::Metadata, "metal", "metadata", "metadata");
  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.verify().valid());

  const auto model = make_relu_model();
  const auto envelope = compiler::CacheEnvelopeBuilder{}.build(
      executable, make_test_cache_options(*model));
  ASSERT_TRUE(envelope.verify(executable).valid());

  auto stale_manifest = envelope;
  stale_manifest.key.manifest_hash = "stale-manifest";
  EXPECT_FALSE(stale_manifest.verify(executable).valid());
  EXPECT_TRUE(has_diagnostic_containing(
      stale_manifest.verify(executable).diagnostics, "manifest hash drift"));

  auto missing_driver = envelope;
  missing_driver.key.driver_identity.clear();
  EXPECT_FALSE(missing_driver.verify(executable).valid());
  EXPECT_TRUE(has_diagnostic_containing(
      missing_driver.verify(executable).diagnostics, "driver identity"));

  auto stale_kernel_unit = envelope;
  stale_kernel_unit.key.kernel_unit_versions.front() = "stale-kernel-unit";
  EXPECT_FALSE(stale_kernel_unit.verify(executable).valid());
  EXPECT_TRUE(has_diagnostic_containing(
      stale_kernel_unit.verify(executable).diagnostics,
      "kernel unit versions drift"));
}

TEST_F(GfxBackendArchitectureContractTest,
       UnknownBackendCannotMaterializeTargetOrStageFactory) {
  EXPECT_ANY_THROW(compiler::BackendTarget::from_backend(GpuBackend::Unknown));
  EXPECT_ANY_THROW(GpuStageFactory::factory_for_backend(GpuBackend::Unknown));
  EXPECT_ANY_THROW(memory_ops_for_backend(GpuBackend::Unknown));
}

TEST_F(GfxBackendArchitectureContractTest,
       KernelRegistriesRequireExplicitOpOwnedUnits) {
  const auto opencl_registry = test::KernelRegistryContract::for_opencl();
  ASSERT_TRUE(opencl_registry.audit_is_valid());

  const auto opencl_common_unit =
      opencl_registry.resolve_unit(LoweringRouteKind::Common, "common");
  ASSERT_TRUE(opencl_common_unit.valid());
  EXPECT_EQ(opencl_common_unit.kind(), KernelUnitKind::Common);
  EXPECT_EQ(opencl_common_unit.op_family(), "common_io");
  const auto opencl_metadata_unit =
      opencl_registry.resolve_unit(LoweringRouteKind::Metadata, "metadata");
  ASSERT_TRUE(opencl_metadata_unit.valid());
  EXPECT_EQ(opencl_metadata_unit.kind(), KernelUnitKind::Metadata);
  EXPECT_EQ(opencl_metadata_unit.op_family(), "metadata");
  EXPECT_TRUE(
      opencl_registry.rejects_unit(LoweringRouteKind::GeneratedKernel, ""));
  EXPECT_TRUE(opencl_registry.rejects_unit(LoweringRouteKind::GeneratedKernel,
                                           "opencl_generated_kernel"));
  const auto matmul_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/matmul_f32");
  ASSERT_TRUE(matmul_unit.valid());
  EXPECT_EQ(matmul_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(matmul_unit.backend_domain(), "opencl");
  EXPECT_EQ(matmul_unit.op_family(), "MatMul");
  const auto shapeof_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/shapeof_i64");
  ASSERT_TRUE(shapeof_unit.valid());
  EXPECT_EQ(shapeof_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(shapeof_unit.backend_domain(), "opencl");
  EXPECT_EQ(shapeof_unit.op_family(), "ShapeOf");
  EXPECT_FALSE(shapeof_unit.exception_contract().valid());
  const auto range_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/range_f32");
  ASSERT_TRUE(range_unit.valid());
  EXPECT_EQ(range_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(range_unit.backend_domain(), "opencl");
  EXPECT_EQ(range_unit.op_family(), "Range");
  EXPECT_FALSE(range_unit.exception_contract().valid());
  const auto dynamic_range_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel,
      "opencl/generated/range_i64_unit_dynamic");
  ASSERT_TRUE(dynamic_range_unit.valid());
  EXPECT_EQ(dynamic_range_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(dynamic_range_unit.backend_domain(), "opencl");
  EXPECT_EQ(dynamic_range_unit.op_family(), "Range");
  EXPECT_FALSE(dynamic_range_unit.exception_contract().valid());
  const auto tile_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/tile_f32");
  ASSERT_TRUE(tile_unit.valid());
  EXPECT_EQ(tile_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(tile_unit.backend_domain(), "opencl");
  EXPECT_EQ(tile_unit.op_family(), "Tile");
  EXPECT_FALSE(tile_unit.exception_contract().valid());
  const auto eltwise_unit =
      opencl_registry.resolve_unit(LoweringRouteKind::GeneratedKernel,
                                   "opencl/generated/eltwise_binary_f32");
  ASSERT_TRUE(eltwise_unit.valid());
  EXPECT_EQ(eltwise_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(eltwise_unit.backend_domain(), "opencl");
  EXPECT_EQ(eltwise_unit.op_family(), "Eltwise");
  const auto logical_bool_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel,
      "opencl/generated/eltwise_logical_binary_bool");
  ASSERT_TRUE(logical_bool_unit.valid());
  EXPECT_EQ(logical_bool_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(logical_bool_unit.backend_domain(), "opencl");
  EXPECT_EQ(logical_bool_unit.op_family(), "Eltwise");
  EXPECT_FALSE(logical_bool_unit.exception_contract().valid());
  const auto compare_unit =
      opencl_registry.resolve_unit(LoweringRouteKind::GeneratedKernel,
                                   "opencl/generated/eltwise_compare_f32");
  ASSERT_TRUE(compare_unit.valid());
  EXPECT_EQ(compare_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(compare_unit.backend_domain(), "opencl");
  EXPECT_EQ(compare_unit.op_family(), "Eltwise");
  EXPECT_FALSE(compare_unit.exception_contract().valid());
  const auto select_unit =
      opencl_registry.resolve_unit(LoweringRouteKind::GeneratedKernel,
                                   "opencl/generated/eltwise_select_f32");
  ASSERT_TRUE(select_unit.valid());
  EXPECT_EQ(select_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(select_unit.backend_domain(), "opencl");
  EXPECT_EQ(select_unit.op_family(), "Eltwise");
  EXPECT_FALSE(select_unit.exception_contract().valid());
  const auto activation_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/activation_f32");
  ASSERT_TRUE(activation_unit.valid());
  EXPECT_EQ(activation_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(activation_unit.backend_domain(), "opencl");
  EXPECT_EQ(activation_unit.op_family(), "Activation");
  const auto runtime_beta_activation_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel,
      "opencl/generated/activation_runtime_beta_f32");
  ASSERT_TRUE(runtime_beta_activation_unit.valid());
  EXPECT_EQ(runtime_beta_activation_unit.kind(),
            KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(runtime_beta_activation_unit.backend_domain(), "opencl");
  EXPECT_EQ(runtime_beta_activation_unit.op_family(), "Activation");
  const auto pool_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/pool2d_f32");
  ASSERT_TRUE(pool_unit.valid());
  EXPECT_EQ(pool_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(pool_unit.backend_domain(), "opencl");
  EXPECT_EQ(pool_unit.op_family(), "Pooling");
  const auto logical_reduce_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/reduction_bool");
  ASSERT_TRUE(logical_reduce_unit.valid());
  EXPECT_EQ(logical_reduce_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(logical_reduce_unit.backend_domain(), "opencl");
  EXPECT_EQ(logical_reduce_unit.op_family(), "Reduction");
  EXPECT_FALSE(logical_reduce_unit.exception_contract().valid());
  const auto transpose_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/transpose_f32");
  ASSERT_TRUE(transpose_unit.valid());
  EXPECT_EQ(transpose_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(transpose_unit.backend_domain(), "opencl");
  EXPECT_EQ(transpose_unit.op_family(), "Transpose");
  EXPECT_FALSE(transpose_unit.exception_contract().valid());
  const auto split_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/split3_f32");
  ASSERT_TRUE(split_unit.valid());
  EXPECT_EQ(split_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(split_unit.backend_domain(), "opencl");
  EXPECT_EQ(split_unit.op_family(), "Split");
  EXPECT_FALSE(split_unit.exception_contract().valid());
  const auto concat_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/concat2_f32");
  ASSERT_TRUE(concat_unit.valid());
  EXPECT_EQ(concat_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(concat_unit.backend_domain(), "opencl");
  EXPECT_EQ(concat_unit.op_family(), "Concat");
  EXPECT_FALSE(concat_unit.exception_contract().valid());

  const auto metal_registry = test::KernelRegistryContract::for_metal();
  ASSERT_TRUE(metal_registry.audit_is_valid());
  const auto metal_common_unit =
      metal_registry.resolve_unit(LoweringRouteKind::Common, "common");
  ASSERT_TRUE(metal_common_unit.valid());
  EXPECT_EQ(metal_common_unit.kind(), KernelUnitKind::Common);
  EXPECT_EQ(metal_common_unit.op_family(), "common_io");
  const auto metal_metadata_unit =
      metal_registry.resolve_unit(LoweringRouteKind::Metadata, "metadata");
  ASSERT_TRUE(metal_metadata_unit.valid());
  EXPECT_EQ(metal_metadata_unit.kind(), KernelUnitKind::Metadata);
  EXPECT_EQ(metal_metadata_unit.op_family(), "metadata");
  EXPECT_TRUE(
      metal_registry.rejects_unit(LoweringRouteKind::GeneratedKernel, ""));
  EXPECT_TRUE(
      metal_registry.rejects_unit(LoweringRouteKind::VendorPrimitive, ""));
  EXPECT_TRUE(metal_registry.rejects_unit(LoweringRouteKind::GeneratedKernel,
                                          "metal_lowering"));
  EXPECT_TRUE(metal_registry.rejects_unit(LoweringRouteKind::VendorPrimitive,
                                          "metal_lowering"));
  EXPECT_TRUE(metal_registry
                  .resolve_unit(LoweringRouteKind::VendorPrimitive,
                                "metal/vendor/mps_gemm")
                  .valid());
  EXPECT_TRUE(metal_registry
                  .resolve_unit(LoweringRouteKind::GeneratedKernel,
                                "metal/generated/slice")
                  .valid());
  const auto metal_eltwise_unit = metal_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "metal/generated/eltwise");
  ASSERT_TRUE(metal_eltwise_unit.valid());
  EXPECT_EQ(metal_eltwise_unit.op_family(), "Eltwise");
  const auto metal_activation_unit = metal_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "metal/generated/activation");
  ASSERT_TRUE(metal_activation_unit.valid());
  EXPECT_EQ(metal_activation_unit.op_family(), "Activation");
  const auto metal_transpose_unit = metal_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "metal/generated/transpose_f32");
  ASSERT_TRUE(metal_transpose_unit.valid());
  EXPECT_EQ(metal_transpose_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(metal_transpose_unit.backend_domain(), "metal");
  EXPECT_EQ(metal_transpose_unit.op_family(), "Transpose");
  const auto metal_pool_unit = metal_registry.resolve_unit(
      LoweringRouteKind::VendorPrimitive, "metal/vendor/mps_pool2d");
  ASSERT_TRUE(metal_pool_unit.valid());
  EXPECT_EQ(metal_pool_unit.op_family(), "Pooling");
}

TEST_F(GfxBackendArchitectureContractTest,
       MetalUnsupportedCoverageNeverSelectsGenericKernelUnit) {
  const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                             ov::Shape{2, 3});
  const auto convert =
      std::make_shared<ov::op::v0::Convert>(input, ov::element::f16);
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_metal_operation_support_policy());

  const auto support = capabilities.query_operation({convert});
  EXPECT_FALSE(support.semantic_legal);
  EXPECT_EQ(support.preferred_route_kind, LoweringRouteKind::Unsupported);
  EXPECT_NE(support.preferred_route, "backend_lowering");
  EXPECT_TRUE(support.semantic_reason == "missing_metal_explicit_kernel_unit" ||
              support.semantic_reason == "unsupported_by_metal_capabilities");
}

TEST_F(GfxBackendArchitectureContractTest,
       KernelRegistryAuditRejectsUndocumentedHandwrittenExceptions) {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  compiler::KernelRegistry registry{
      target,
      {compiler::KernelUnit::describe(
          LoweringRouteKind::HandwrittenKernelException,
          KernelUnitKind::HandwrittenException, "opencl/baseline/undocumented",
          target.backend_id(), "Eltwise")}};

  const auto audit = registry.audit();
  EXPECT_FALSE(audit.valid());
  EXPECT_EQ(audit.handwritten_exception_count, 1u);
  bool found_contract_diagnostic = false;
  for (const auto &diagnostic : audit.diagnostics) {
    if (diagnostic.find("missing exception contract") != std::string::npos) {
      found_contract_diagnostic = true;
    }
  }
  EXPECT_TRUE(found_contract_diagnostic);
}

TEST_F(GfxBackendArchitectureContractTest,
       ExecutablePayloadRoutesRequireCompilerMaterializedPayloads) {
  struct RouteCase {
    LoweringRouteKind route_kind;
    std::string backend_domain;
    std::string kernel_unit_id;
    std::string kernel_unit_kind;
  };

  const std::vector<RouteCase> routes = {
      {LoweringRouteKind::GeneratedKernel, "metal", "metal/generated/test",
       "generated_kernel"},
      {LoweringRouteKind::GeneratedKernel, "opencl", "opencl/generated/test",
       "generated_kernel"},
      {LoweringRouteKind::VendorPrimitive, "metal", "metal/vendor/test",
       "vendor_primitive"},
  };

  for (const auto &route : routes) {
    const auto manifest = make_single_payload_route_manifest(
        route.route_kind, route.backend_domain, route.kernel_unit_id,
        route.kernel_unit_kind);
    ASSERT_TRUE(manifest.valid()) << route.kernel_unit_id;

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    const auto executable_result = executable.verify();
    EXPECT_FALSE(executable_result.valid()) << route.kernel_unit_id;
    EXPECT_TRUE(has_diagnostic_containing(executable_result.diagnostics,
                                          "requires a materialized payload"))
        << route.kernel_unit_id;

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    const auto runtime_result = compiler::verify_runtime_executable_descriptor(
        runtime_descriptor, executable);
    EXPECT_FALSE(runtime_result.valid()) << route.kernel_unit_id;
    EXPECT_TRUE(has_diagnostic_containing(runtime_result.diagnostics,
                                          "requires a materialized payload"))
        << route.kernel_unit_id;
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       OpenClPayloadMaterializationIsOwnedByBackendModule) {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy());
  const compiler::OperationLegalizer legalizer(capabilities);
  const compiler::LoweringPlanner planner(
      target, compiler::make_opencl_kernel_registry(target));

  const auto lowering_plan = planner.plan(make_relu_model(), legalizer);
  ASSERT_TRUE(lowering_plan.executable());
  const auto manifest = compiler::ManifestBuilder{}.build(lowering_plan);
  ASSERT_TRUE(manifest.valid());

  const auto common_executable =
      compiler::ExecutableBundleBuilder{}.build(manifest, lowering_plan);
  EXPECT_TRUE(common_executable.artifact_payloads.empty());
  EXPECT_FALSE(common_executable.verify().valid());
  EXPECT_TRUE(has_diagnostic_containing(common_executable.verify().diagnostics,
                                        "requires a materialized payload"));

  const auto backend_executable =
      compiler::ExecutableBundleBuilder(
          compiler::make_opencl_kernel_artifact_payload_resolver())
          .build(manifest, lowering_plan);
  ASSERT_TRUE(backend_executable.verify().valid());
  ASSERT_EQ(backend_executable.artifact_payloads.size(), 1u);

  const auto descriptor_index =
      backend_executable.artifact_payloads.front().artifact_descriptor_index;
  ASSERT_LT(descriptor_index, backend_executable.artifact_descriptors.size());
  const auto &artifact =
      backend_executable.artifact_descriptors[descriptor_index];
  EXPECT_EQ(artifact.kernel.kernel_id, "opencl/generated/activation_f32");
  EXPECT_EQ(artifact.payload_kind,
            compiler::KernelArtifactPayloadKind::OpenClSource);
  EXPECT_EQ(artifact.entry_point, "gfx_opencl_generated_activation_f32");
}

TEST_F(GfxBackendArchitectureContractTest,
       OpenClRangePayloadUsesRegisteredOpOwnedKernelUnit) {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy());
  const compiler::OperationLegalizer legalizer(capabilities);
  const compiler::LoweringPlanner planner(
      target, compiler::make_opencl_kernel_registry(target));

  const auto lowering_plan = planner.plan(make_static_range_model(), legalizer);
  ASSERT_TRUE(lowering_plan.executable());

  bool found_range = false;
  for (const auto &op : lowering_plan.operations) {
    if (op.type_name != "Range") {
      continue;
    }
    found_range = true;
    EXPECT_EQ(op.kernel_unit.route_kind(), LoweringRouteKind::GeneratedKernel);
    EXPECT_EQ(op.kernel_unit.kind(), KernelUnitKind::GeneratedKernel);
    EXPECT_EQ(op.kernel_unit.backend_domain(), "opencl");
    EXPECT_EQ(op.kernel_unit.op_family(), "Range");
    EXPECT_EQ(op.kernel_unit.id(), "opencl/generated/range_f32");
  }
  ASSERT_TRUE(found_range);

  const auto manifest = compiler::ManifestBuilder{}.build(lowering_plan);
  ASSERT_TRUE(manifest.valid());
  const auto backend_executable =
      compiler::ExecutableBundleBuilder(
          compiler::make_opencl_kernel_artifact_payload_resolver())
          .build(manifest, lowering_plan);
  ASSERT_TRUE(backend_executable.verify().valid());
  ASSERT_EQ(backend_executable.artifact_payloads.size(), 1u);

  const auto descriptor_index =
      backend_executable.artifact_payloads.front().artifact_descriptor_index;
  ASSERT_LT(descriptor_index, backend_executable.artifact_descriptors.size());
  const auto &artifact =
      backend_executable.artifact_descriptors[descriptor_index];
  EXPECT_EQ(artifact.kernel.kernel_id, "opencl/generated/range_f32");
  EXPECT_EQ(artifact.payload_kind,
            compiler::KernelArtifactPayloadKind::OpenClSource);
  EXPECT_EQ(artifact.entry_point, "gfx_opencl_generated_range_f32");
  ASSERT_TRUE(backend_executable.artifact_payloads.front().payload);
  EXPECT_EQ(backend_executable.artifact_payloads.front().payload->source_id(),
            "opencl/generated/range_f32");
}

TEST_F(GfxBackendArchitectureContractTest,
       RegisteredBackendModulesShareTheSamePassthroughContract) {
  const auto contracts = backend_catalog.compiled_module_contracts();
  ASSERT_FALSE(contracts.empty());

  for (const auto &module_contract : contracts) {
    const auto model =
        models.passthrough(ov::PartialShape{ov::Dimension::dynamic(), 3});
    const auto compile_result =
        module_contract.compile_without_graph_pipeline(model);
    EXPECT_TRUE(
        module_contract.compile_result_obeys_manifest_contract(compile_result))
        << module_contract.target().debug_string() << ": "
        << compile_result.unsupported_message();
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RegisteredBackendModulesOwnGraphPipelineOptions) {
  const auto contracts = backend_catalog.compiled_module_contracts();
  ASSERT_FALSE(contracts.empty());

  for (const auto &module_contract : contracts) {
    const auto &module = module_contract.module();
    const auto &options = module.pipeline_options();
    EXPECT_EQ(&options, &module.pipeline_options())
        << module_contract.target().debug_string();

    if (module_contract.target().backend() == GpuBackend::Metal) {
      EXPECT_TRUE(options.preserve_scaled_dot_product_attention);
      EXPECT_TRUE(options.canonicalize_sigmoid_before_ranking);
      EXPECT_TRUE(options.enable_llm_attention_fusions);
    }
    if (module_contract.target().backend() == GpuBackend::OpenCL) {
      EXPECT_FALSE(options.preserve_scaled_dot_product_attention);
      EXPECT_FALSE(options.canonicalize_sigmoid_before_ranking);
      EXPECT_FALSE(options.enable_llm_attention_fusions);
    }
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RegisteredBackendModulesOwnFusionCapabilities) {
  const auto contracts = backend_catalog.compiled_module_contracts();
  ASSERT_FALSE(contracts.empty());

  for (const auto &module_contract : contracts) {
    const auto &fusion = module_contract.module().capabilities().fusion();
    if (module_contract.target().backend() == GpuBackend::Metal) {
      EXPECT_FALSE(fusion.enable_generic_attention_fusion);
      EXPECT_TRUE(fusion.supports_vendor_attention_stage);
      EXPECT_TRUE(fusion.enable_conv_activation_fusion);
      EXPECT_FALSE(fusion.enable_precision_sensitive_arithmetic_fusion);
    }
    if (module_contract.target().backend() == GpuBackend::OpenCL) {
      EXPECT_TRUE(fusion.enable_generic_attention_fusion);
      EXPECT_FALSE(fusion.supports_vendor_attention_stage);
      EXPECT_TRUE(fusion.enable_conv_activation_fusion);
      EXPECT_TRUE(fusion.enable_precision_sensitive_arithmetic_fusion);
    }
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RegisteredBackendModulesOwnPostOpFusionCapabilities) {
  const auto contracts = backend_catalog.compiled_module_contracts();
  ASSERT_FALSE(contracts.empty());

  for (const auto &module_contract : contracts) {
    const auto &capabilities = module_contract.module().capabilities();
    const auto &post_ops = capabilities.post_ops();
    if (module_contract.target().backend() == GpuBackend::Metal) {
      EXPECT_TRUE(post_ops.allow_stage_bias_fusion("Convolution"));
      EXPECT_TRUE(post_ops.allow_stage_bias_fusion("GroupConvolution"));
      EXPECT_TRUE(post_ops.allow_stage_batchnorm_fusion("Convolution"));
      EXPECT_TRUE(post_ops.allow_stage_batchnorm_fusion("GroupConvolution"));
      EXPECT_TRUE(capabilities.allow_stage_activation_fusion(
          "Convolution", ActivationKind::Relu));
      EXPECT_TRUE(capabilities.allow_stage_activation_fusion(
          "GroupConvolution", ActivationKind::Swish));
      EXPECT_FALSE(capabilities.allow_stage_activation_fusion(
          "Convolution", ActivationKind::Gelu));
    }
    if (module_contract.target().backend() == GpuBackend::OpenCL) {
      EXPECT_TRUE(post_ops.allow_stage_bias_fusion("Convolution"));
      EXPECT_FALSE(post_ops.allow_stage_bias_fusion("GroupConvolution"));
      EXPECT_TRUE(post_ops.allow_stage_batchnorm_fusion("Convolution"));
      EXPECT_FALSE(post_ops.allow_stage_batchnorm_fusion("GroupConvolution"));
      EXPECT_TRUE(capabilities.allow_stage_activation_fusion(
          "Convolution", ActivationKind::Relu));
      EXPECT_TRUE(capabilities.allow_stage_activation_fusion(
          "Convolution", ActivationKind::Swish));
      EXPECT_FALSE(capabilities.allow_stage_activation_fusion(
          "Convolution", ActivationKind::Sigmoid));
      EXPECT_FALSE(capabilities.allow_stage_activation_fusion(
          "GroupConvolution", ActivationKind::Relu));
    }
  }
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
