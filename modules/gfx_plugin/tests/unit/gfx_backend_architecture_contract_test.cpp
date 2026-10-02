// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_backend_architecture_contract_common.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

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

static_assert(std::is_same_v<decltype(&GpuStageFactory::create),
                             std::unique_ptr<GpuStage> (*)(
                                 const RuntimeStageMaterializationContext &,
                                 GpuBackend, void *, void *)>,
              "GpuStageFactory must require the unified compiler-owned stage "
              "materialization context");

static_assert(
    !std::is_default_constructible_v<RuntimeStageMaterializationContext>,
    "Runtime stage materialization context must not allow a null/default "
    "descriptor escape hatch");

static_assert(
    std::is_constructible_v<RuntimeStageMaterializationContext,
                            const RuntimeStageExecutableDescriptor &>,
    "Runtime stage materialization context must be constructed from the "
    "compiler-owned stage descriptor");

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

static_assert(std::is_same_v<decltype(&create_view_only_stage),
                             std::unique_ptr<GpuStage> (*)(
                                 const RuntimeStageExecutableDescriptor &)>,
              "Shared view-only runtime stage materialization must use the "
              "compiler-owned "
              "stage descriptor, not an ov::Node");

static_assert(
    std::is_same_v<decltype(&PipelineStageMaterializer::create_stage),
                   std::unique_ptr<GpuStage> (PipelineStageMaterializer::*)(
                       const RuntimeStageExecutableDescriptor &) const>,
    "PipelineStageMaterializer must materialize stages from compiler-owned "
    "runtime descriptors without an ov::Node bridge");

static_assert(
    std::is_same_v<
        decltype(&PipelineStageMaterializer::descriptor_for_stage_index),
        const RuntimeStageExecutableDescriptor *(
            PipelineStageMaterializer::*)(size_t) const noexcept>,
    "PipelineStageMaterializer must expose descriptor-owned runtime stage "
    "indices without fallback repair");

static_assert(
    std::is_same_v<
        decltype(RuntimeExecutableDescriptor::materialization_stages),
        std::vector<PipelineStageMaterializationPlan>>,
    "Runtime executable descriptor must own compiler materialization stages "
    "instead of referencing a parallel stage plan");

static_assert(
    std::is_same_v<decltype(RuntimeExecutableDescriptor::runtime_options),
                   PipelineStageRuntimeOptionsPlan>,
    "Runtime executable descriptor must carry compiler runtime options "
    "without a separate runtime plan object");

static_assert(
    std::is_same_v<
        decltype(RuntimeExecutableDescriptor::materialization_finalized), bool>,
    "Runtime executable descriptor must explicitly distinguish a seed "
    "descriptor from a finalized compiler materialization contract");

using RuntimeShapeOfPlannerFn =
    RuntimeValuePlan (*)(const RuntimeInputResolver &, std::string_view);
using RuntimeBroadcastPlannerFn = RuntimeValuePlan (*)(
    const RuntimeInputResolver &, const ov::Shape &, std::string_view);
using RuntimeRangePlannerFn = RuntimeValuePlan (*)(const RuntimeInputResolver &,
                                                   std::string_view);
using RuntimeSelectPlannerFn =
    RuntimeSelectPlan (*)(const RuntimeInputResolver &, std::string_view);
using RuntimeReshapePlannerFn = RuntimeValuePlan (*)(
    const RuntimeInputResolver &, const RuntimeStageExecutableDescriptor &,
    std::string_view);
using RuntimeSqueezePlannerFn = RuntimeValuePlan (*)(
    const RuntimeInputResolver &, const RuntimeStageExecutableDescriptor &,
    std::string_view);
using RuntimeShapePreservingPlannerFn = RuntimeValuePlan (*)(
    const RuntimeInputResolver &, const RuntimeStageExecutableDescriptor &,
    std::string_view);
using RuntimeConvertPlannerFn = RuntimeValuePlan (*)(
    const RuntimeInputResolver &, const RuntimeStageExecutableDescriptor &,
    std::string_view);
using RuntimeGatherPlannerFn = RuntimeGatherPlan (*)(
    const RuntimeInputResolver &, const RuntimeStageExecutableDescriptor &,
    std::string_view);
using RuntimeScatterUpdatePlannerFn = RuntimeScatterUpdatePlan (*)(
    const RuntimeInputResolver &, const RuntimeStageExecutableDescriptor &,
    std::string_view);
using RuntimeSplitPlannerFn = RuntimeSplitPlan (*)(
    const RuntimeInputResolver &, const RuntimeStageExecutableDescriptor &,
    size_t, std::string_view);
using RuntimeTransposePlannerFn = RuntimeTransposePlan (*)(
    const RuntimeInputResolver &, const RuntimeStageExecutableDescriptor &,
    std::string_view);
using RuntimeInterpolatePlannerFn = RuntimeInterpolatePlan (*)(
    const RuntimeInputResolver &, const std::vector<GpuTensor *> &,
    const RuntimeStageExecutableDescriptor &, std::string_view);
using RuntimeReducePlannerFn =
    RuntimeReducePlan (*)(const RuntimeInputResolver &, std::string_view,
                          const RuntimeReduceInfo &, std::string_view);
using RuntimeReduceDescriptorInfoFn = std::optional<RuntimeReduceInfo> (*)(
    const RuntimeStageExecutableDescriptor &, const ov::Shape &,
    std::string_view);
using RuntimeReduceDescriptorDispatchFn = RuntimeReduceDispatchPlan (*)(
    const RuntimeStageExecutableDescriptor &, std::string_view);
using RuntimeConcatPlannerFn =
    RuntimeConcatPlan (*)(const RuntimeInputResolver &, std::string_view);
using RuntimeSlicePlannerFn = RuntimeSlicePlan (*)(
    const RuntimeInputResolver &, const std::vector<GpuTensor *> &, bool,
    std::string_view);
using RuntimeSmallI64BinderFn = bool (*)(GpuBufferManager *,
                                         const std::vector<GpuTensor *> &,
                                         std::vector<GpuTensor> &,
                                         GfxProfiler *, bool, std::string_view,
                                         std::string_view);
using DescriptorRuntimeShapeRuleFn = bool (*)(std::string_view,
                                              std::string_view) noexcept;
using RuntimeShapeMaterializationRuleForFn =
    std::optional<RuntimeShapeMaterializationRule> (*)(
        std::string_view) noexcept;
using RuntimeShapeMaterializationRuleSupportedFn =
    bool (*)(std::string_view) noexcept;
using RuntimeShapeMaterializerFn = RuntimeShapeMaterializationResult (*)(
    const RuntimeShapeMaterializationRequest &);

static_assert(std::is_same_v<decltype(static_cast<RuntimeShapeOfPlannerFn>(
                                 &plan_shapeof_runtime_values)),
                             RuntimeShapeOfPlannerFn>,
              "Shared runtime ShapeOf planning must be descriptor/input based");

static_assert(
    std::is_same_v<decltype(static_cast<RuntimeBroadcastPlannerFn>(
                       &plan_broadcast_runtime_values)),
                   RuntimeBroadcastPlannerFn>,
    "Shared runtime Broadcast planning must be descriptor/input based");

static_assert(std::is_same_v<decltype(static_cast<RuntimeRangePlannerFn>(
                                 &plan_range_runtime_values)),
                             RuntimeRangePlannerFn>,
              "Shared runtime Range planning must be descriptor/input based");

static_assert(std::is_same_v<decltype(static_cast<RuntimeSelectPlannerFn>(
                                 &plan_select_runtime_values)),
                             RuntimeSelectPlannerFn>,
              "Shared runtime Select planning must be descriptor/input based");

static_assert(std::is_same_v<decltype(static_cast<RuntimeReshapePlannerFn>(
                                 &plan_reshape_runtime_values)),
                             RuntimeReshapePlannerFn>,
              "Shared runtime Reshape planning must be descriptor/input based");

static_assert(
    std::is_same_v<decltype(static_cast<RuntimeSqueezePlannerFn>(
                       &plan_squeeze_unsqueeze_runtime_values)),
                   RuntimeSqueezePlannerFn>,
    "Shared runtime Squeeze/Unsqueeze planning must be descriptor/input based");

static_assert(
    std::is_same_v<decltype(static_cast<RuntimeShapePreservingPlannerFn>(
                       &plan_shape_preserving_runtime_values)),
                   RuntimeShapePreservingPlannerFn>,
    "Shared runtime shape-preserving planning must be descriptor/input based");

static_assert(std::is_same_v<decltype(static_cast<RuntimeConvertPlannerFn>(
                                 &plan_convert_runtime_values)),
                             RuntimeConvertPlannerFn>,
              "Shared runtime Convert planning must be descriptor/input based");

static_assert(std::is_same_v<decltype(static_cast<RuntimeGatherPlannerFn>(
                                 &plan_gather_runtime_values)),
                             RuntimeGatherPlannerFn>,
              "Shared runtime Gather planning must be descriptor/input based");

static_assert(
    std::is_same_v<decltype(static_cast<RuntimeScatterUpdatePlannerFn>(
                       &plan_scatter_update_runtime_values)),
                   RuntimeScatterUpdatePlannerFn>,
    "Shared runtime ScatterUpdate planning must be descriptor/input based");

static_assert(std::is_same_v<decltype(static_cast<RuntimeSplitPlannerFn>(
                                 &plan_split_runtime_values)),
                             RuntimeSplitPlannerFn>,
              "Shared runtime Split planning must be descriptor/input based");

static_assert(
    std::is_same_v<decltype(static_cast<RuntimeTransposePlannerFn>(
                       &plan_transpose_runtime_values)),
                   RuntimeTransposePlannerFn>,
    "Shared runtime Transpose planning must be descriptor/input based");

static_assert(
    std::is_same_v<decltype(static_cast<RuntimeInterpolatePlannerFn>(
                       &plan_interpolate_runtime_values)),
                   RuntimeInterpolatePlannerFn>,
    "Shared runtime Interpolate planning must be descriptor/input based");

static_assert(std::is_same_v<decltype(static_cast<RuntimeReducePlannerFn>(
                                 &plan_reduce_runtime_values)),
                             RuntimeReducePlannerFn>,
              "Shared runtime Reduce planning must be descriptor/input based");

static_assert(
    std::is_same_v<decltype(static_cast<RuntimeReduceDescriptorInfoFn>(
                       &runtime_reduce_info_from_descriptor)),
                   RuntimeReduceDescriptorInfoFn>,
    "Shared runtime Reduce metadata must come from the runtime descriptor");

static_assert(
    std::is_same_v<decltype(static_cast<RuntimeReduceDescriptorDispatchFn>(
                       &runtime_reduce_dispatch_from_descriptor)),
                   RuntimeReduceDescriptorDispatchFn>,
    "Shared runtime Reduce dispatch must come from descriptor launch metadata");

static_assert(std::is_same_v<decltype(static_cast<RuntimeConcatPlannerFn>(
                                 &plan_concat_runtime_values)),
                             RuntimeConcatPlannerFn>,
              "Shared runtime Concat planning must be descriptor/input based");

static_assert(std::is_same_v<decltype(static_cast<RuntimeSlicePlannerFn>(
                                 &plan_slice_runtime_values)),
                             RuntimeSlicePlannerFn>,
              "Shared runtime Slice planning must be descriptor/input based");

static_assert(
    !std::is_invocable_v<RuntimeSlicePlannerFn, const RuntimeInputResolver &,
                         const std::vector<GpuTensor *> &, const ov::Node *,
                         bool, std::string_view>,
    "Shared runtime Slice planner must not accept ov::Node");

static_assert(
    std::is_same_v<decltype(static_cast<RuntimeSmallI64BinderFn>(
                       &bind_small_i64_const_stage_outputs)),
                   RuntimeSmallI64BinderFn>,
    "Shared runtime small i64 const binding must not require ov::Node output "
    "metadata");

static_assert(
    !std::is_invocable_v<RuntimeShapeOfPlannerFn, const RuntimeInputResolver &,
                         const ov::Node *, std::string_view>,
    "Shared runtime ShapeOf planner must not accept ov::Node");

static_assert(
    std::is_same_v<decltype(static_cast<DescriptorRuntimeShapeRuleFn>(
                       &descriptor_owns_runtime_shape_rule)),
                   DescriptorRuntimeShapeRuleFn>,
    "Descriptor-owned runtime shape routing must be expressed as a shared "
    "op-family/rule contract, not as ov::Node runtime inspection");

static_assert(
    std::is_same_v<decltype(static_cast<RuntimeShapeMaterializationRuleForFn>(
                       &runtime_shape_materialization_rule_for)),
                   RuntimeShapeMaterializationRuleForFn>,
    "Runtime shape materialization must be selected from descriptor runtime "
    "shape rule vocabulary, not from backend, op-family, or ov::Node "
    "inspection");

static_assert(
    std::is_same_v<
        decltype(static_cast<RuntimeShapeMaterializationRuleSupportedFn>(
            &runtime_shape_materialization_rule_supported)),
        RuntimeShapeMaterializationRuleSupportedFn>,
    "Runtime shape materialization support must be backend-neutral and keyed "
    "only by descriptor runtime shape rule metadata");

static_assert(
    std::is_same_v<decltype(static_cast<RuntimeShapeMaterializerFn>(
                       &materialize_runtime_output_shapes)),
                   RuntimeShapeMaterializerFn>,
    "Runtime shape materialization must consume one descriptor-owned request "
    "and return one typed materialization result");

static_assert(
    !std::is_invocable_v<RuntimeShapeMaterializerFn, GpuBackend,
                         const RuntimeShapeMaterializationRequest &>,
    "Runtime shape materialization must not branch on backend identity");

static_assert(
    !std::is_invocable_v<RuntimeShapeMaterializationRuleForFn, GpuBackend,
                         std::string_view, std::string_view>,
    "Runtime shape materialization rule lookup must not branch on backend");

static_assert(
    !std::is_invocable_v<RuntimeShapeMaterializationRuleForFn, std::string_view,
                         std::string_view>,
    "Runtime shape materialization rule lookup must not duplicate compiler "
    "op-family ownership checks");

static_assert(
    !std::is_invocable_v<RuntimeReducePlannerFn, const RuntimeInputResolver &,
                         const ov::Node *, std::string_view,
                         const RuntimeReduceInfo &, std::string_view>,
    "Shared runtime Reduce planner must not accept ov::Node");

static_assert(!std::is_invocable_v<RuntimeReduceDescriptorInfoFn,
                                   const std::shared_ptr<const ov::Node> &,
                                   const ov::Shape &, std::string_view>,
              "Shared runtime Reduce metadata helper must not accept ov::Node");

static_assert(
    !std::is_invocable_v<RuntimeRangePlannerFn, const RuntimeInputResolver &,
                         const ov::Node *, std::string_view>,
    "Shared runtime Range planner must not accept ov::Node");

static_assert(
    !std::is_invocable_v<RuntimeReshapePlannerFn, const RuntimeInputResolver &,
                         const ov::Node &, std::string_view>,
    "Shared runtime Reshape planner must not accept ov::Node");

static_assert(
    !std::is_invocable_v<RuntimeSqueezePlannerFn, const RuntimeInputResolver &,
                         const ov::Node &, std::string_view>,
    "Shared runtime Squeeze/Unsqueeze planner must not accept ov::Node");

static_assert(
    !std::is_invocable_v<RuntimeGatherPlannerFn, const RuntimeInputResolver &,
                         const ov::Node &, std::string_view>,
    "Shared runtime Gather planner must not accept ov::Node");

static_assert(!std::is_invocable_v<RuntimeSplitPlannerFn, const ov::Node *,
                                   const ov::Shape &, size_t, std::string_view>,
              "Shared runtime Split planner must not accept ov::Node");

static_assert(
    !std::is_invocable_v<
        RuntimeInterpolatePlannerFn, const RuntimeInputResolver &,
        const std::vector<GpuTensor *> &, const ov::Node &, std::string_view>,
    "Shared runtime Interpolate planner must not accept ov::Node");

static_assert(
    !std::is_invocable_v<RuntimeSmallI64BinderFn, GpuBufferManager *,
                         const std::vector<GpuTensor *> &,
                         std::vector<GpuTensor> &,
                         const std::shared_ptr<const ov::Node> &, GfxProfiler *,
                         bool, std::string_view, std::string_view>,
    "Shared runtime small i64 const binding must not accept ov::Node");

using RuntimeExecutableDescriptorFinalBuildFn = RuntimeExecutableDescriptor (
    compiler::RuntimeExecutableDescriptorBuilder::*)(
    const compiler::RuntimeExecutableDescriptorBuildRequest &) const;

using PipelineStageGraphSnapshotBuildFn =
    compiler::detail::PipelineStageGraphSnapshot (*)(
        const std::shared_ptr<const ov::Model> &, const FusionConfig &);

static_assert(
    std::is_same_v<
        decltype(&compiler::detail::make_pipeline_stage_graph_snapshot),
        PipelineStageGraphSnapshotBuildFn>,
    "Pipeline stage graph snapshot must be a compiler-owned contract produced "
    "before runtime stage descriptor materialization");

using PipelineStageFusionConfigBuildFn =
    FusionConfig (*)(const compiler::FusionCapabilities &, bool, bool);

static_assert(
    std::is_same_v<
        decltype(&compiler::detail::make_pipeline_stage_fusion_config),
        PipelineStageFusionConfigBuildFn>,
    "Pipeline stage fusion config must be derived from backend capabilities in "
    "compiler detail code, not from runtime/backend branches");

static_assert(
    std::is_same_v<decltype(compiler::detail::PipelineStageBuildRequest::graph),
                   compiler::detail::PipelineStageGraphSnapshot>,
    "Pipeline stage runtime descriptor builder must consume a graph snapshot "
    "instead of rebuilding graph ownership from runtime state");

using PipelineStageRuntimeDescriptorBuildFn = RuntimeExecutableDescriptor (*)(
    const compiler::detail::PipelineStageBuildRequest &);

static_assert(
    std::is_same_v<
        decltype(&compiler::detail::build_pipeline_stage_runtime_descriptor),
        PipelineStageRuntimeDescriptorBuildFn>,
    "Runtime descriptor stage materialization must have one compiler detail "
    "entry point shared by all backend targets");

static_assert(
    std::is_same_v<decltype(&compiler::RuntimeExecutableDescriptorBuilder::
                                build_finalized),
                   RuntimeExecutableDescriptorFinalBuildFn>,
    "RuntimeExecutableDescriptorBuilder must consume the explicit compiler "
    "stage graph snapshot request and own final stage "
    "materialization/descriptor verification for all backend targets");

using RuntimeExecutableDescriptorBuildRequestGraphSnapshotMember =
    const compiler::detail::PipelineStageGraphSnapshot
        *compiler::RuntimeExecutableDescriptorBuildRequest::*;

static_assert(
    std::is_same_v<decltype(&compiler::RuntimeExecutableDescriptorBuildRequest::
                                stage_graph_snapshot),
                   RuntimeExecutableDescriptorBuildRequestGraphSnapshotMember>,
    "RuntimeExecutableDescriptorBuildRequest must carry the compiler-owned "
    "stage graph snapshot separately from the immutable ExecutableBundle");

using RuntimeExecutableDescriptorBuildRequestValidFn = bool (
    compiler::RuntimeExecutableDescriptorBuildRequest::*)() const noexcept;

static_assert(
    std::is_same_v<
        decltype(&compiler::RuntimeExecutableDescriptorBuildRequest::valid),
        RuntimeExecutableDescriptorBuildRequestValidFn>,
    "RuntimeExecutableDescriptorBuildRequest must expose a backend-neutral "
    "validity contract for executable, graph snapshot, registry and target");

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
                       const compiler::BackendTarget &, const ov::AnyMap &,
                       const ov::SoPtr<ov::IRemoteContext> &)>,
    "BackendRuntimeProvider must create runtime state for a concrete "
    "compiler BackendTarget, not for a backend enum alone");

static_assert(
    std::is_same_v<decltype(BackendRuntimeProvider::execute_infer),
                   void (*)(InferRequest &,
                            const std::shared_ptr<const CompiledModel> &)>,
    "BackendRuntimeProvider must expose one narrow infer execution callback "
    "instead of requiring common InferRequest backend switches");

static_assert(
    !std::is_invocable_v<decltype(&create_backend_state), GpuBackend,
                         const ov::AnyMap &,
                         const ov::SoPtr<ov::IRemoteContext> &>,
    "Runtime state creation must not accept GpuBackend-only identity");

static_assert(
    !std::is_invocable_v<decltype(&execute_backend_infer), GpuBackend,
                         InferRequest &,
                         const std::shared_ptr<const CompiledModel> &>,
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
       DescriptorLaunchPlanRolesUseSharedAbiMapping) {
  KernelLaunchPlanDescriptor descriptor;
  descriptor.valid = true;
  descriptor.buffer_roles = {
      std::string(
          kernel_buffer_role_descriptor_name(GfxKernelBufferRole::TensorInput)),
      std::string(
          kernel_buffer_role_descriptor_name(GfxKernelBufferRole::ConstTensor)),
      std::string(kernel_buffer_role_descriptor_name(
          GfxKernelBufferRole::TensorOutput)),
      std::string(
          kernel_buffer_role_descriptor_name(GfxKernelBufferRole::ScalarParam)),
      std::string(kernel_buffer_role_descriptor_name(
          GfxKernelBufferRole::RuntimeParams)),
  };

  const auto roles =
      materialize_descriptor_launch_roles(descriptor, "unit_launch_roles");
  EXPECT_EQ(
      roles,
      (std::vector<GfxKernelBufferRole>{
          GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::ConstTensor,
          GfxKernelBufferRole::TensorOutput, GfxKernelBufferRole::ScalarParam,
          GfxKernelBufferRole::RuntimeParams}));

  descriptor.buffer_roles[1] = "unknown_backend_local_role";
  EXPECT_THROW((void)materialize_descriptor_launch_roles(descriptor,
                                                         "unit_launch_roles"),
               ov::Exception);
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorLaunchPlanMaterializesRuntimeBindingWithoutNodeFallback) {
  RuntimeStageExecutableDescriptor descriptor;
  descriptor.stage_name = "descriptor_owned_binding";
  descriptor.entry_point = "descriptor_owned_entry";
  descriptor.abi_arg_count = 5;
  descriptor.runtime_param_i64_metadata = {1, 2, 3};
  descriptor.runtime_param_reduce_keep_dims = true;
  descriptor.runtime_param_reduce_keep_dims_valid = true;
  descriptor.launch_plan.valid = true;
  descriptor.launch_plan.buffer_roles = {
      std::string(
          kernel_buffer_role_descriptor_name(GfxKernelBufferRole::TensorInput)),
      std::string(
          kernel_buffer_role_descriptor_name(GfxKernelBufferRole::ConstTensor)),
      std::string(kernel_buffer_role_descriptor_name(
          GfxKernelBufferRole::TensorOutput)),
      std::string(
          kernel_buffer_role_descriptor_name(GfxKernelBufferRole::ScalarParam)),
      std::string(kernel_buffer_role_descriptor_name(
          GfxKernelBufferRole::RuntimeParams)),
  };
  descriptor.launch_plan.direct_input_indices = {2};
  descriptor.launch_plan.scalar_args = {42};

  const auto binding =
      make_stage_descriptor_kernel_runtime_binding(descriptor, "unit_binding");

  EXPECT_EQ(binding.inputs, (std::vector<size_t>{2}));
  EXPECT_EQ(binding.input_arg_count, 3u);
  EXPECT_EQ(binding.operand_kinds, (std::vector<int32_t>{1, 1, 1, 0, 1}));
  EXPECT_EQ(binding.operand_arg_indices,
            (std::vector<int32_t>{0, 1, 3, -1, 2}));
  EXPECT_EQ(binding.scalar_args, (std::vector<int32_t>{42}));
  EXPECT_EQ(binding.runtime_param_i64_metadata,
            (std::vector<int64_t>{1, 2, 3}));
  EXPECT_TRUE(binding.runtime_param_reduce_keep_dims);
  EXPECT_TRUE(binding.runtime_param_reduce_keep_dims_valid);

  descriptor.launch_plan.direct_input_indices = {0, 1};
  EXPECT_THROW((void)make_stage_descriptor_kernel_runtime_binding(
                   descriptor, "unit_binding_bad_inputs"),
               ov::Exception);
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeStageMaterializationContextUsesDescriptorIdentityOnly) {
  RuntimeStageExecutableDescriptor descriptor;
  descriptor.op_family = "DescriptorOwnedOp";
  descriptor.stage_name = "descriptor_owned_stage";
  descriptor.manifest_ref = "manifest://unit/descriptor_owned_stage";
  descriptor.kernel_id = "unit/descriptor_owned_stage";

  RuntimeStageMaterializationContext context{descriptor};

  EXPECT_EQ(context.op_type_name(), std::string("DescriptorOwnedOp"));
  EXPECT_EQ(context.op_friendly_name(), std::string("descriptor_owned_stage"));
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

} // namespace
} // namespace gfx_plugin
} // namespace ov
