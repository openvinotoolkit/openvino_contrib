// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
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
#include "kernel_ir/gfx_kernel_source.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "mlir/gfx_stage_kernel_binding.hpp"
#include "mlir/mlir_stage_runtime_value_bridge.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"
#include "runtime/backend_request_state.hpp"
#include "runtime/backend_runtime.hpp"
#include "runtime/backend_runtime_provider.hpp"
#include "runtime/backend_stage_factory.hpp"
#include "runtime/descriptor_const_tensor_materializer.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/execution_dispatcher.hpp"
#include "runtime/gfx_stage_runtime_values.hpp"
#include "runtime/gpu_buffer.hpp"
#include "runtime/gpu_buffer_manager.hpp"
#include "runtime/gpu_memory_ops.hpp"
#include "runtime/kernel_launch_plan.hpp"
#include "runtime/pipeline_stage_materializer.hpp"
#include "runtime/pipeline_stage_plan.hpp"
#include "runtime/runtime_session.hpp"
#include "runtime/stateful_stage.hpp"
#include "runtime/tensor_binding_contract.hpp"
#include "runtime/view_only_stage.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
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
    std::is_same_v<decltype(RuntimeExecutableDescriptor::materialization_finalized),
                   bool>,
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

  static_assert(std::is_same_v<decltype(static_cast<RuntimeTransposePlannerFn>(
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
      !std::is_invocable_v<RuntimeReshapePlannerFn,
                           const RuntimeInputResolver &, const ov::Node &,
                           std::string_view>,
      "Shared runtime Reshape planner must not accept ov::Node");

  static_assert(
      !std::is_invocable_v<RuntimeSqueezePlannerFn,
                           const RuntimeInputResolver &, const ov::Node &,
                           std::string_view>,
      "Shared runtime Squeeze/Unsqueeze planner must not accept ov::Node");

  static_assert(
      !std::is_invocable_v<RuntimeGatherPlannerFn, const RuntimeInputResolver &,
                           const ov::Node &, std::string_view>,
      "Shared runtime Gather planner must not accept ov::Node");

  static_assert(
      !std::is_invocable_v<RuntimeSplitPlannerFn, const ov::Node *,
                           const ov::Shape &, size_t, std::string_view>,
      "Shared runtime Split planner must not accept ov::Node");

  static_assert(
      !std::is_invocable_v<RuntimeInterpolatePlannerFn,
                           const RuntimeInputResolver &,
                           const std::vector<GpuTensor *> &, const ov::Node &,
                           std::string_view>,
      "Shared runtime Interpolate planner must not accept ov::Node");

static_assert(
    !std::is_invocable_v<RuntimeSmallI64BinderFn, GpuBufferManager *,
                         const std::vector<GpuTensor *> &,
                         std::vector<GpuTensor> &,
                         const std::shared_ptr<const ov::Node> &, GfxProfiler *,
                         bool, std::string_view, std::string_view>,
    "Shared runtime small i64 const binding must not accept ov::Node");

using RuntimeExecutableDescriptorFinalBuildFn =
    RuntimeExecutableDescriptor (
        compiler::RuntimeExecutableDescriptorBuilder::*)(
        const compiler::RuntimeExecutableDescriptorBuildRequest &) const;

static_assert(
    std::is_same_v<
        decltype(&compiler::RuntimeExecutableDescriptorBuilder::build_finalized),
        RuntimeExecutableDescriptorFinalBuildFn>,
    "RuntimeExecutableDescriptorBuilder must own final graph snapshot, stage "
    "materialization and descriptor verification for all backend targets");

static_assert(
    std::is_same_v<decltype(&compiler::build_pipeline_stage_runtime_descriptor),
                   RuntimeExecutableDescriptor (*)(
                       const compiler::PipelineStageBuildRequest &)>,
    "Compiler pipeline stage builder must return the final runtime descriptor "
    "instead of exposing a parallel runtime-facing stage plan");

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

class UnitMetadataBufferManager final : public GpuBufferManager {
public:
  GpuBuffer wrap_const(const std::string &, const void *, size_t bytes,
                       ov::element::Type type) override {
    GpuBuffer buffer;
    buffer.buffer = reinterpret_cast<GpuBufferHandle>(m_next_handle);
    buffer.size = bytes;
    buffer.type = type;
    buffer.persistent = true;
    buffer.allocation_uid = allocate_gpu_buffer_uid();
    m_next_handle += 0x1000u;
    return buffer;
  }

private:
  uintptr_t m_next_handle = 0x100000u;
};

class UnitDescriptorConstBufferManager final : public GpuBufferManager {
public:
  struct Upload {
    std::string key;
    std::vector<uint8_t> bytes;
    ov::element::Type type;
    uint64_t allocation_uid = 0;
  };

  bool supports_const_cache() const override { return true; }

  GpuBuffer wrap_const(const std::string &key, const void *data, size_t bytes,
                       ov::element::Type type) override {
    Upload upload;
    upload.key = key;
    upload.bytes.resize(bytes);
    if (bytes != 0) {
      std::memcpy(upload.bytes.data(), data, bytes);
    }
    upload.type = type;

    GpuBuffer buffer;
    buffer.buffer = reinterpret_cast<GpuBufferHandle>(m_next_handle);
    buffer.size = bytes;
    buffer.type = type;
    buffer.persistent = true;
    buffer.allocation_uid = allocate_gpu_buffer_uid();
    upload.allocation_uid = buffer.allocation_uid;
    m_next_handle += 0x1000u;
    uploads.push_back(std::move(upload));
    return buffer;
  }

  std::vector<Upload> uploads;

private:
  uintptr_t m_next_handle = 0x200000u;
};

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

  bool valid() const noexcept override { return true; }
};

class UnitNoopStage final : public GpuStage {
public:
  UnitNoopStage(std::string name, std::string type)
      : m_name(std::move(name)), m_type(std::move(type)) {}

  void init(GpuBufferManager *) override {}
  void prepare_runtime_handle(GpuBufferManager *) override {}
  void execute(GpuCommandBufferHandle) override {}
  void set_inputs(const std::vector<GpuTensor *> &inputs) override {
    m_inputs = inputs;
  }
  void set_output(GpuTensor *output) override { m_output = output; }
  const std::string &name() const override { return m_name; }
  const std::string &type() const override { return m_type; }
  std::unique_ptr<GpuStage> clone() const override {
    auto stage = std::make_unique<UnitNoopStage>(m_name, m_type);
    stage->m_inputs = m_inputs;
    stage->m_output = m_output;
    return stage;
  }

private:
  std::string m_name;
  std::string m_type;
  std::vector<GpuTensor *> m_inputs;
  GpuTensor *m_output = nullptr;
};

class UnitBackendStageFactory final : public BackendStageFactory {
public:
  GpuBackend backend() const override { return GpuBackend::Metal; }

  std::unique_ptr<GpuStage>
  create_stage(const RuntimeStageMaterializationContext &) const override {
    return nullptr;
  }
};

class CapturingBackendStageFactory final : public BackendStageFactory {
public:
  GpuBackend backend() const override { return GpuBackend::Metal; }

  std::unique_ptr<GpuStage> create_stage(
      const RuntimeStageMaterializationContext &context) const override {
    stage_names.push_back(context.op_friendly_name());

    const auto &descriptor = context.require_descriptor();
    if (auto stateful = create_stateful_stage(descriptor)) {
      return stateful;
    }
    if (auto view = create_view_only_stage(descriptor)) {
      return view;
    }
    return std::make_unique<UnitNoopStage>(context.op_friendly_name(),
                                           context.op_type_name());
  }

  mutable std::vector<std::string> stage_names;
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

RuntimeStageExecutableDescriptor
make_materializer_base_descriptor(const std::shared_ptr<ov::Node> &node) {
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

PipelineStageMaterializationPlan
make_vendor_materialization_plan(const std::shared_ptr<ov::Node> &input,
                                 const std::shared_ptr<ov::Node> &node,
                                 RuntimeStageExecutableDescriptor descriptor) {
  (void)input;
  descriptor.payload_kind = KernelArtifactPayloadKind::VendorDescriptor;
  descriptor.payload = std::make_shared<UnitVendorPayload>();
  descriptor.artifact_key = "artifact://unit/vendor_attention";

  PipelineStageMaterializationPlan plan;
  plan.kind = PipelineStageMaterializationKind::VendorAttention;
  plan.io_plan.stage_name =
      node ? node->get_friendly_name() : "unit_vendor_attention";
  plan.io_plan.op_family = node ? node->get_type_name() : "VendorAttention";
  plan.io_plan.runtime_stage_index = 0;
  plan.descriptor_stage_index = descriptor.stage_index;
  PipelineStageInputLink input_link;
  input_link.port = 0;
  input_link.source_ref.kind = PipelineStageTensorRefKind::Parameter;
  input_link.source_ref.index = 0;
  input_link.source_ref.port = 0;
  plan.io_plan.inputs.push_back(input_link);

  PipelineStageOutputDesc output;
  output.shape = ov::Shape{1};
  output.type = ov::element::f32;
  output.source_port = 0;
  output.source_ref.kind = PipelineStageTensorRefKind::StageOutput;
  output.source_ref.index = 0;
  output.source_ref.port = 0;
  plan.io_plan.outputs.push_back(std::move(output));

  plan.vendor_attention.name = "unit_vendor_attention";
  plan.vendor_attention.descriptor = descriptor;
  plan.materialized_descriptor = std::move(descriptor);
  plan.materialized_descriptor_valid = true;
  return plan;
}

PipelineStageMaterializationPlan
make_single_materialization_plan(const std::shared_ptr<const ov::Node> &node,
                                 RuntimeStageExecutableDescriptor descriptor) {
  PipelineStageMaterializationPlan plan;
  plan.kind = PipelineStageMaterializationKind::SingleStage;
  plan.io_plan.stage_name =
      node ? node->get_friendly_name() : std::string("unit_single_stage");
  plan.io_plan.op_family =
      node ? node->get_type_name() : std::string("Unknown");
  plan.io_plan.runtime_stage_index = descriptor.stage_index;
  plan.descriptor_stage_index = descriptor.stage_index;

  PipelineStageOutputDesc output;
  output.shape = ov::Shape{1};
  output.type = ov::element::f32;
  output.source_port = 0;
  output.source_ref.kind = PipelineStageTensorRefKind::StageOutput;
  output.source_ref.index = descriptor.stage_index;
  output.source_ref.port = 0;
  plan.io_plan.outputs.push_back(std::move(output));

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
  EXPECT_EQ(binding.operand_kinds,
            (std::vector<int32_t>{1, 1, 1, 0, 1}));
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
  view_descriptor.output_bindings.push_back(make_runtime_binding(
      "reshape.output0", "reshape_output", "TensorOutput"));
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

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStageMaterializerMaterializesDescriptorOnly) {
  auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape{2, 3});
  auto shape =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {6});
  auto reshape = std::make_shared<ov::op::v1::Reshape>(parameter, shape, false);
  auto relu = std::make_shared<ov::op::v0::Relu>(parameter);

  auto view_descriptor = make_materializer_base_descriptor(reshape);
  view_descriptor.origin = KernelArtifactOrigin::Metadata;
  view_descriptor.payload_kind = KernelArtifactPayloadKind::None;
  view_descriptor.kernel_id = "metadata";
  view_descriptor.entry_point = "metadata";
  view_descriptor.layout_contract = "view_only";
  view_descriptor.tensor_view_only = true;
  view_descriptor.input_bindings = {make_runtime_binding(
      "reshape.input0", "compiler_view_input_region", "TensorInput")};
  view_descriptor.output_bindings = {make_runtime_binding(
      "reshape.output0", "compiler_view_output_region", "TensorOutput")};
  view_descriptor.output_bindings.front().partial_shape = "{6}";
  view_descriptor.abi_arg_count = 1;
  view_descriptor.abi_output_arg_count = 1;

  auto payload_descriptor = make_materializer_base_descriptor(relu);
  payload_descriptor.stage_index = 1;
  payload_descriptor.stage_record_key = 0x2234u;
  payload_descriptor.manifest_ref = "manifest://unit/relu_payload";
  payload_descriptor.abi_fingerprint = "abi://unit/relu_payload";
  payload_descriptor.artifact_key = "artifact://unit/relu_payload";

  RuntimeExecutableDescriptor runtime_descriptor;
  runtime_descriptor.target_fingerprint = "metal:unit";
  runtime_descriptor.stages = {view_descriptor, payload_descriptor};
  runtime_descriptor.materialization_finalized = true;

  auto view_plan = make_single_materialization_plan(reshape, view_descriptor);
  view_plan.io_plan.outputs.front().shape = ov::Shape{6};
  runtime_descriptor.materialization_stages.push_back(std::move(view_plan));
  runtime_descriptor.materialization_stages.push_back(
      make_single_materialization_plan(relu, payload_descriptor));

  CapturingBackendStageFactory stage_factory;
  PipelineStageRuntimeMaterializationRequest request;
  request.stage_factory = &stage_factory;
  request.runtime_descriptor = &runtime_descriptor;

  auto pipeline = materialize_pipeline_stage_descriptors(request);
  ASSERT_EQ(pipeline.size(), 2u);
  ASSERT_EQ(stage_factory.stage_names.size(), 2u);
  EXPECT_EQ(stage_factory.stage_names[0], reshape->get_friendly_name());
  EXPECT_EQ(stage_factory.stage_names[1], relu->get_friendly_name());
}

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStageMaterializerProjectsDescriptorOutputContracts) {
  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{8});
  auto relu = std::make_shared<ov::op::v0::Relu>(parameter);

  auto descriptor = make_materializer_base_descriptor(relu);
  descriptor.origin = KernelArtifactOrigin::Metadata;
  descriptor.payload_kind = KernelArtifactPayloadKind::None;
  descriptor.kernel_id = "metadata";
  descriptor.entry_point = "metadata";
  descriptor.output_bindings.front().element_type = "i32";
  descriptor.output_bindings.front().partial_shape = "{4,2}";

  PipelineStageMaterializationPlan plan;
  plan.kind = PipelineStageMaterializationKind::SingleStage;
  plan.io_plan.stage_name = relu->get_friendly_name();
  plan.io_plan.op_family = relu->get_type_name();
  plan.io_plan.runtime_stage_index = descriptor.stage_index;
  plan.descriptor_stage_index = descriptor.stage_index;

  PipelineStageOutputDesc output;
  output.type = ov::element::dynamic;
  output.source_port = 0;
  output.source_ref.kind = PipelineStageTensorRefKind::StageOutput;
  output.source_ref.index = descriptor.stage_index;
  output.source_ref.port = 0;
  plan.io_plan.outputs.push_back(std::move(output));
  plan.materialized_descriptor = descriptor;
  plan.materialized_descriptor_valid = true;

  RuntimeExecutableDescriptor runtime_descriptor;
  runtime_descriptor.target_fingerprint = "metal:unit";
  runtime_descriptor.stages = {descriptor};
  runtime_descriptor.materialization_finalized = true;

  runtime_descriptor.materialization_stages.push_back(std::move(plan));

  CapturingBackendStageFactory stage_factory;
  PipelineStageRuntimeMaterializationRequest request;
  request.stage_factory = &stage_factory;
  request.runtime_descriptor = &runtime_descriptor;

  auto pipeline = materialize_pipeline_stage_descriptors(request);
  ASSERT_EQ(pipeline.size(), 1u);
  ASSERT_EQ(pipeline.front().outputs.size(), 1u);
  EXPECT_EQ(pipeline.front().outputs.front().type, ov::element::i32);
  EXPECT_EQ(pipeline.front().outputs.front().shape, (ov::Shape{4, 2}));
  ASSERT_EQ(stage_factory.stage_names.size(), 1u);
  EXPECT_EQ(stage_factory.stage_names.front(), relu->get_friendly_name());
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

compiler::TensorContract
make_source_payload_tensor_contract(compiler::TensorContractRole role,
                                    size_t index, std::string shape) {
  auto contract = make_tensor_contract(role);
  const bool input = role == compiler::TensorContractRole::TensorInput;
  contract.logical_name = (input ? "input" : "output") + std::to_string(index);
  contract.memory_region_id = "stage_0." + contract.logical_name + "_region";
  contract.partial_shape = std::move(shape);
  return contract;
}

compiler::ManifestBundle make_source_payload_route_manifest(
    std::string backend_domain, std::string op_family,
    const std::vector<std::string> &input_shapes,
    const std::vector<std::string> &output_shapes) {
  compiler::StageRecord stage;
  stage.stage_id = 0;
  stage.stable_record_key = 0x1234u;
  stage.source_node_name = op_family;
  stage.normalized_op_family = std::move(op_family);
  stage.execution_kind = LoweringRouteKind::GeneratedKernel;
  stage.backend_domain = std::move(backend_domain);
  stage.kernel_unit_id = stage.backend_domain + "/generated/unit_source";
  stage.kernel_unit_kind = "generated_kernel";
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    stage.inputs.push_back(make_source_payload_tensor_contract(
        compiler::TensorContractRole::TensorInput, i, input_shapes[i]));
  }
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    stage.outputs.push_back(make_source_payload_tensor_contract(
        compiler::TensorContractRole::TensorOutput, i, output_shapes[i]));
  }
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

GfxKernelStageManifest
make_unit_source_stage_manifest(GfxKernelBackendDomain backend_domain,
                                const std::vector<GfxKernelBufferRole> &roles) {
  GfxKernelStageManifest manifest;
  manifest.valid = true;
  manifest.stage_family = GfxKernelStageFamily::Eltwise;
  manifest.backend_domain = backend_domain;
  manifest.execution_kind = GfxKernelExecutionKind::CustomKernel;
  manifest.storage = GfxKernelStorageKind::Buffer;
  manifest.compute_precision = GfxKernelComputePrecision::Native;
  manifest.custom_kernel.valid = true;
  manifest.custom_kernel.kernel_family = "unit_source";
  manifest.custom_kernel.kernel_family_id = 1;
  manifest.custom_kernel.entry_point = "unit_source_entry";
  manifest.custom_kernel.external_buffer_abi = make_gfx_kernel_roles_abi(roles);
  return manifest;
}

uint32_t
count_runtime_param_roles(const std::vector<GfxKernelBufferRole> &roles) {
  uint32_t count = 0;
  for (const auto role : roles) {
    if (role == GfxKernelBufferRole::RuntimeParams) {
      ++count;
    }
  }
  return count;
}

uint32_t count_kernel_roles(const std::vector<GfxKernelBufferRole> &roles,
                            GfxKernelBufferRole expected) {
  uint32_t count = 0;
  for (const auto role : roles) {
    if (role == expected) {
      ++count;
    }
  }
  return count;
}

std::vector<size_t>
make_unit_direct_input_indices(const std::vector<GfxKernelBufferRole> &roles) {
  std::vector<size_t> indices;
  size_t next_input = 0;
  for (const auto role : roles) {
    if (role == GfxKernelBufferRole::TensorInput) {
      indices.push_back(next_input++);
    }
  }
  return indices;
}

KernelLaunchPlanDescriptor make_unit_launch_plan_descriptor(
    const std::vector<GfxKernelBufferRole> &roles,
    const GfxKernelSourceRuntimeBinding &runtime_binding) {
  KernelLaunchPlanDescriptor plan;
  plan.valid = !roles.empty();
  plan.buffer_roles.reserve(roles.size());
  for (const auto role : roles) {
    plan.buffer_roles.emplace_back(kernel_buffer_role_descriptor_name(role));
  }
  plan.direct_input_indices = make_unit_direct_input_indices(roles);
  plan.input_indices = runtime_binding.inputs;
  plan.input_arg_count = runtime_binding.input_arg_count;
  plan.operand_kinds = runtime_binding.operand_kinds;
  plan.operand_arg_indices = runtime_binding.operand_arg_indices;
  plan.scalar_args = runtime_binding.scalar_args;
  return plan;
}

std::shared_ptr<const KernelArtifactPayload> make_unit_source_payload(
    const compiler::KernelArtifactDescriptor &descriptor,
    const std::vector<GfxKernelBufferRole> &roles,
    const GfxKernelSourceRuntimeBinding &runtime_binding = {}) {
  if (descriptor.payload_kind == KernelArtifactPayloadKind::OpenClSource) {
    GfxOpenClSourceArtifact artifact;
    artifact.valid = true;
    artifact.artifact_ref.valid = true;
    artifact.artifact_ref.kind = GfxKernelArtifactKind::OpenClSource;
    artifact.artifact_ref.backend_domain = GfxKernelBackendDomain::OpenCl;
    artifact.artifact_ref.source_id = descriptor.kernel.kernel_id;
    artifact.artifact_ref.entry_point = descriptor.entry_point;
    artifact.source = "__kernel void unit_source_entry() {}";
    artifact.stage_manifest =
        make_unit_source_stage_manifest(GfxKernelBackendDomain::OpenCl, roles);
    artifact.arg_count = static_cast<uint32_t>(roles.size());
    artifact.direct_input_count =
        count_kernel_roles(roles, GfxKernelBufferRole::TensorInput);
    artifact.direct_output_count =
        count_kernel_roles(roles, GfxKernelBufferRole::TensorOutput);
    artifact.direct_input_indices = make_unit_direct_input_indices(roles);
    return std::make_shared<GfxOpenClSourceArtifactPayload>(
        std::move(artifact));
  }

  return std::make_shared<GfxKernelSourcePayload>(
      descriptor.kernel.kernel_id, descriptor.kernel.backend_domain,
      descriptor.entry_point, GfxKernelSourceLanguage::MetalShadingLanguage,
      "kernel void unit_source_entry() {}",
      make_unit_source_stage_manifest(GfxKernelBackendDomain::AppleMsl, roles),
      runtime_binding);
}

compiler::ExecutableBundle make_source_payload_executable(
    std::string backend_domain, std::string op_family,
    const std::vector<GfxKernelBufferRole> &roles,
    const std::vector<std::string> &input_shapes,
    const std::vector<std::string> &output_shapes,
    const GfxKernelSourceRuntimeBinding &runtime_binding = {},
    std::vector<KernelArtifactConstTensor> const_tensors = {}) {
  auto manifest = make_source_payload_route_manifest(
      std::move(backend_domain), std::move(op_family), input_shapes,
      output_shapes);
  auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  OPENVINO_ASSERT(executable.artifact_descriptors.size() == 1,
                  "unit source payload executable must have one descriptor");
  auto &descriptor = executable.artifact_descriptors.front();
  descriptor.entry_point = "unit_source_entry";
  descriptor.abi_arg_count = static_cast<uint32_t>(roles.size());
  descriptor.abi_output_arg_count =
      count_kernel_roles(roles, GfxKernelBufferRole::TensorOutput);
  descriptor.runtime_param_buffer_count = count_runtime_param_roles(roles);
  descriptor.runtime_param_i64_metadata =
      runtime_binding.runtime_param_i64_metadata;
  descriptor.runtime_param_reduce_keep_dims =
      runtime_binding.runtime_param_reduce_keep_dims;
  descriptor.runtime_param_reduce_keep_dims_valid =
      runtime_binding.runtime_param_reduce_keep_dims_valid;
  descriptor.launch_plan =
      make_unit_launch_plan_descriptor(roles, runtime_binding);
  compiler::finalize_kernel_artifact_descriptor_identity(descriptor);
  compiler::KernelArtifactPayloadRecord payload_record;
  payload_record.artifact_descriptor_index = 0;
  payload_record.artifact_key = descriptor.artifact_key;
  payload_record.payload =
      make_unit_source_payload(descriptor, roles, runtime_binding);
  payload_record.const_tensors = std::move(const_tensors);
  executable.artifact_payloads.push_back(std::move(payload_record));
  return executable;
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
  ov::disable_constant_folding(range);
  auto result = std::make_shared<ov::op::v0::Result>(range);
  return std::make_shared<ov::Model>(ov::ResultVector{result},
                                     ov::ParameterVector{});
}

std::shared_ptr<ov::Model> make_add_constant_model(float value) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3});
  auto constant = ov::op::v0::Constant::create(
      ov::element::f32, ov::Shape{1, 3}, {value, value, value});
  auto add = std::make_shared<ov::op::v1::Add>(input, constant);
  auto result = std::make_shared<ov::op::v0::Result>(add);
  return std::make_shared<ov::Model>(ov::ResultVector{result},
                                     ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> make_reshape_model(bool special_zero) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{2, 3});
  auto pattern =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
  auto reshape =
      std::make_shared<ov::op::v1::Reshape>(input, pattern, special_zero);
  auto result = std::make_shared<ov::op::v0::Result>(reshape);
  return std::make_shared<ov::Model>(ov::ResultVector{result},
                                     ov::ParameterVector{input});
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
    EXPECT_TRUE(
        fingerprints.insert(target_contract.target().fingerprint()).second)
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

TEST_F(
    GfxBackendArchitectureContractTest,
    BackendCapabilitiesDoNotAdvertiseCompiledModelExportImportBeforeEnvelopeRoundTrip) {
  const auto module_contracts = backend_catalog.compiled_module_contracts();
  ASSERT_FALSE(module_contracts.empty());
  for (const auto &module_contract : module_contracts) {
    const auto &capabilities = module_contract.module().capabilities();
    EXPECT_FALSE(
        capabilities.artifact_formats().supports_compiled_model_export_import)
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
  if (const auto metal = registry.resolve(
          compiler::BackendTarget::from_backend(GpuBackend::Metal))) {
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

  if (const auto opencl = registry.resolve(
          compiler::BackendTarget::from_backend(GpuBackend::OpenCL))) {
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
  EXPECT_FALSE(stage_build_request.graph.valid());

  compiler::RuntimeExecutableDescriptorBuildRequest descriptor_build_request;
  EXPECT_EQ(descriptor_build_request.target.backend(), GpuBackend::Unknown);
  EXPECT_FALSE(descriptor_build_request.valid());

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
       CompilerMemoryPlanUsesProducerConsumerLifetimesAcrossTargets) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 2, 3});
  auto relu0 = std::make_shared<ov::op::v0::Relu>(input);
  auto relu1 = std::make_shared<ov::op::v0::Relu>(relu0);
  auto relu2 = std::make_shared<ov::op::v0::Relu>(relu1);
  auto result = std::make_shared<ov::op::v0::Result>(relu2);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                           ov::ParameterVector{input});
  (void)model;

  struct TargetCase {
    std::string name;
    compiler::BackendTarget target;
  };
  const std::vector<TargetCase> targets = {
      {"macos",
       compiler::BackendTarget::from_backend(GpuBackend::Metal)},
      {"android",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno)},
      {"rpi4",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D)},
      {"rpi5",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D)}};

  for (const auto &target : targets) {
    SCOPED_TRACE(target.name);
    compiler::LoweringPlan lowering_plan;
    lowering_plan.target = target.target;
    lowering_plan.operations.push_back(make_metadata_planned_operation(
        relu0, compiler::select_tensor_layout_plan("Relu", relu0)));
    lowering_plan.operations.push_back(make_metadata_planned_operation(
        relu1, compiler::select_tensor_layout_plan("Relu", relu1)));
    lowering_plan.operations.push_back(make_metadata_planned_operation(
        relu2, compiler::select_tensor_layout_plan("Relu", relu2)));

    const auto manifest = compiler::ManifestBuilder{}.build(lowering_plan);
    ASSERT_TRUE(manifest.verify().valid());
    ASSERT_EQ(manifest.stages.size(), 3u);

    const auto &memory_plan = manifest.memory_plan;
    const auto &stage0_output = manifest.stages[0].outputs.front();
    const auto &stage1_input = manifest.stages[1].inputs.front();
    const auto &stage1_output = manifest.stages[1].outputs.front();
    const auto &stage2_output = manifest.stages[2].outputs.front();
    EXPECT_EQ(stage1_input.memory_region_id, stage0_output.memory_region_id);

    const auto *stage0_region =
        memory_plan.find_region(stage0_output.memory_region_id);
    const auto *stage1_region =
        memory_plan.find_region(stage1_output.memory_region_id);
    const auto *stage2_region =
        memory_plan.find_region(stage2_output.memory_region_id);
    ASSERT_NE(stage0_region, nullptr);
    ASSERT_NE(stage1_region, nullptr);
    ASSERT_NE(stage2_region, nullptr);
    EXPECT_EQ(stage0_region->kind, compiler::MemoryRegionKind::TransientTensor);
    EXPECT_EQ(stage1_region->kind, compiler::MemoryRegionKind::TransientTensor);
    EXPECT_EQ(stage2_region->kind, compiler::MemoryRegionKind::TransientTensor);
    EXPECT_EQ(stage0_region->lifetime.first_stage, 0u);
    EXPECT_EQ(stage0_region->lifetime.last_stage, 1u);
    EXPECT_EQ(stage1_region->lifetime.first_stage, 1u);
    EXPECT_EQ(stage1_region->lifetime.last_stage, 2u);
    EXPECT_EQ(stage2_region->lifetime.first_stage, 2u);
    EXPECT_EQ(stage2_region->lifetime.last_stage, 2u);
    EXPECT_EQ(stage0_region->alias_group, stage2_region->alias_group);
    EXPECT_NE(stage0_region->alias_group, stage1_region->alias_group);
    EXPECT_FALSE(memory_plan.hidden_host_copies_allowed);
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerMemoryPlanClassifiesExternalAndModelOwnedInputsAcrossTargets) {
  const auto model = make_add_constant_model(2.0f);
  std::shared_ptr<const ov::Node> add;
  for (const auto &node : model->get_ordered_ops()) {
    if (ov::as_type_ptr<const ov::op::v1::Add>(node)) {
      add = node;
      break;
    }
  }
  ASSERT_TRUE(add);

  struct TargetCase {
    std::string name;
    compiler::BackendTarget target;
  };
  const std::vector<TargetCase> targets = {
      {"macos",
       compiler::BackendTarget::from_backend(GpuBackend::Metal)},
      {"android",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno)},
      {"rpi4",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D)},
      {"rpi5",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D)}};

  for (const auto &target : targets) {
    SCOPED_TRACE(target.name);
    compiler::LoweringPlan lowering_plan;
    lowering_plan.target = target.target;
    auto add_operation = make_metadata_planned_operation(
        add, compiler::select_tensor_layout_plan("Add", add));
    add_operation.input_element_types = {"f32", "f32"};
    add_operation.input_shapes = {"{1,3}", "{1,3}"};
    lowering_plan.operations.push_back(std::move(add_operation));

    const auto manifest = compiler::ManifestBuilder{}.build(lowering_plan);
    ASSERT_TRUE(manifest.verify().valid());
    ASSERT_EQ(manifest.stages.size(), 1u);
    ASSERT_EQ(manifest.stages.front().inputs.size(), 2u);
    const auto &parameter_input = manifest.stages.front().inputs[0];
    const auto &constant_input = manifest.stages.front().inputs[1];
    const auto *parameter_region =
        manifest.memory_plan.find_region(parameter_input.memory_region_id);
    const auto *constant_region =
        manifest.memory_plan.find_region(constant_input.memory_region_id);
    ASSERT_NE(parameter_region, nullptr);
    ASSERT_NE(constant_region, nullptr);

    EXPECT_EQ(parameter_region->kind, compiler::MemoryRegionKind::ExternalTensor);
    EXPECT_TRUE(parameter_region->external_binding);
    EXPECT_EQ(parameter_input.lifetime_class, "external");
    EXPECT_EQ(constant_region->kind, compiler::MemoryRegionKind::ImmutableTensor);
    EXPECT_FALSE(constant_region->external_binding);
    EXPECT_EQ(constant_input.lifetime_class, "model_owned");
    EXPECT_FALSE(manifest.memory_plan.hidden_host_copies_allowed);
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerMemoryPlanRejectsOverlappingTransientAliasGroups) {
  compiler::MemoryPlan plan;
  plan.schema_version = 1;

  compiler::MemoryRegion first;
  first.region_id = "stage_0.output_0";
  first.logical_tensor_name = "relu0.output0";
  first.kind = compiler::MemoryRegionKind::TransientTensor;
  first.element_type = "f32";
  first.partial_shape = "{1,2,3}";
  first.layout = "logical";
  first.storage_kind = "device_buffer";
  first.alias_group = "transient_alias_0";
  first.lifetime = {0, 2};

  compiler::MemoryRegion second = first;
  second.region_id = "stage_1.output_0";
  second.logical_tensor_name = "relu1.output0";
  second.lifetime = {1, 3};

  compiler::AliasGroup alias_group;
  alias_group.group_id = "transient_alias_0";
  alias_group.region_ids = {first.region_id, second.region_id};

  compiler::TransientArena arena;
  arena.arena_id = "transient_device_buffer_arena";
  arena.storage_kind = "device_buffer";
  arena.region_ids = alias_group.region_ids;

  plan.regions = {std::move(first), std::move(second)};
  plan.alias_groups = {std::move(alias_group)};
  plan.transient_arenas = {std::move(arena)};

  const auto verification = plan.verify();
  EXPECT_FALSE(verification.valid());
  EXPECT_TRUE(has_diagnostic_containing(
      verification.diagnostics, "overlapping transient lifetimes"));
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

  RuntimeExecutableDescriptor runtime_descriptor;
  runtime_descriptor.target_fingerprint = "metal:unit";
  runtime_descriptor.stages.push_back(make_materializer_base_descriptor(relu));

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
  PipelineStageMaterializer materializer(stage_factory, runtime_descriptor, {});

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

  RuntimeExecutableDescriptor runtime_descriptor;
  runtime_descriptor.target_fingerprint = "metal:unit";
  runtime_descriptor.stages.push_back(make_materializer_base_descriptor(relu));

  auto incomplete_vendor_descriptor = runtime_descriptor.stages.front();
  incomplete_vendor_descriptor.input_bindings.clear();
  incomplete_vendor_descriptor.output_bindings.clear();
  incomplete_vendor_descriptor.abi_arg_count = 0;
  incomplete_vendor_descriptor.abi_output_arg_count = 0;

  const auto plan = make_vendor_materialization_plan(
      parameter, relu, incomplete_vendor_descriptor);
  UnitBackendStageFactory stage_factory;
  PipelineStageMaterializer materializer(stage_factory, runtime_descriptor, {});

  EXPECT_THROW((void)materializer.create_materialized_descriptor(plan),
               ov::Exception);
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeExecutableDescriptorRejectsIncompleteMaterialization) {
  const auto manifest = make_single_payload_route_manifest(
      LoweringRouteKind::Metadata, "opencl", "metadata", "metadata");
  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.valid());

  auto runtime_descriptor =
      compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
  ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(runtime_descriptor,
                                                            executable));

  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto relu = std::make_shared<ov::op::v0::Relu>(parameter);
  PipelineStageMaterializationPlan materialized_stage;
  materialized_stage.kind = PipelineStageMaterializationKind::SingleStage;
  materialized_stage.io_plan.stage_name = relu->get_friendly_name();
  materialized_stage.io_plan.op_family = relu->get_type_name();
  materialized_stage.io_plan.runtime_stage_index = 0;
  materialized_stage.descriptor_stage_index = 0;
  runtime_descriptor.materialization_finalized = true;
  runtime_descriptor.materialization_stages.push_back(
      std::move(materialized_stage));
  const auto unfrozen_verification =
      compiler::verify_runtime_executable_descriptor_materialization(
          runtime_descriptor);
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
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 2, 3});
    auto starts =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto ends =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto steps =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto axes =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto slice =
        std::make_shared<ov::op::v8::Slice>(input, starts, ends, steps, axes);
    op.source_node = slice;
    op.node_name = slice->get_friendly_name();
    op.type_name = slice->get_type_name();
    op.input_element_types = {"f32", "i64", "i64", "i64", "i64"};
    op.input_shapes = {"{1,2,3}", "{1}", "{1}", "{1}", "{1}"};
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
    std::vector<int64_t> expected_i64_metadata;
  };

  const std::vector<int64_t> slice_metadata = {1, 1, 5};
  const std::vector<int64_t> strided_slice_metadata = {
      1, 2, 4, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0};

  for (const auto test_case :
       {Case{"Concat", "concat", {1}}, Case{"Broadcast", "broadcast", {0, 2}},
        Case{"Select", "select", {}}, Case{"ShapeOf", "shape_of", {}},
        Case{"Slice", "slice", slice_metadata},
        Case{"StridedSlice", "slice", strided_slice_metadata},
        Case{"Range", "range", {}}, Case{"Tile", "tile", {}},
        Case{"Relu", "static_or_descriptor", {}}}) {
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
    if (std::string_view(test_case.op_type) == "Concat") {
      auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 2, 3});
      auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 2, 3});
      auto concat =
          std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lhs, rhs}, 1);
      op.source_node = concat;
      op.node_name = concat->get_friendly_name();
      op.type_name = concat->get_type_name();
      op.input_element_types = {"f32", "f32"};
      op.input_shapes = {"{1,2,3}", "{1,2,3}"};
      op.output_shapes = {"{1,4,3}"};
    } else if (std::string_view(test_case.op_type) == "Broadcast") {
      auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape{1, 3});
      auto target =
          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
      auto broadcast = std::make_shared<ov::op::v3::Broadcast>(input, target);
      op.source_node = broadcast;
      op.node_name = broadcast->get_friendly_name();
      op.type_name = broadcast->get_type_name();
      op.input_element_types = {"f32", "i64"};
      op.input_shapes = {"{1,3}", "{2}"};
      op.output_shapes = {"{2,3}"};
    } else if (std::string_view(test_case.op_type) == "Slice") {
      auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape{1, 2, 3});
      auto starts =
          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
      auto ends =
          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
      auto steps =
          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
      auto axes =
          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
      auto slice =
          std::make_shared<ov::op::v8::Slice>(input, starts, ends, steps, axes);
      op.source_node = slice;
      op.node_name = slice->get_friendly_name();
      op.type_name = slice->get_type_name();
      op.input_element_types = {"f32", "i64", "i64", "i64", "i64"};
      op.input_shapes = {"{1,2,3}", "{1}", "{1}", "{1}", "{1}"};
      op.output_shapes = {"{1,2,3}"};
    } else if (std::string_view(test_case.op_type) == "StridedSlice") {
      auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape{1, 2, 3});
      auto begin = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3},
                                                {0, 0, 0});
      auto end = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3},
                                              {1, 2, 3});
      auto strides = ov::op::v0::Constant::create(ov::element::i64,
                                                  ov::Shape{3}, {1, 1, 1});
      const std::vector<int64_t> zero_mask = {0, 0, 0};
      auto slice = std::make_shared<ov::op::v1::StridedSlice>(
          input, begin, end, strides, zero_mask, zero_mask, zero_mask,
          zero_mask, zero_mask);
      op.source_node = slice;
      op.node_name = slice->get_friendly_name();
      op.type_name = slice->get_type_name();
      op.input_element_types = {"f32", "i64", "i64", "i64"};
      op.input_shapes = {"{1,2,3}", "{3}", "{3}", "{3}"};
      op.output_shapes = {"{1,2,3}"};
    }
    plan.operations.push_back(std::move(op));

    const auto manifest = compiler::ManifestBuilder{}.build(plan);
    ASSERT_TRUE(manifest.valid());
    ASSERT_EQ(manifest.stages.size(), 1u);
    EXPECT_EQ(manifest.stages.front().runtime_shape.rule,
              test_case.expected_rule);
    EXPECT_EQ(manifest.stages.front().runtime_shape.i64_metadata,
              test_case.expected_i64_metadata);

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    ASSERT_TRUE(executable.valid());
    ASSERT_EQ(executable.artifact_descriptors.size(), 1u);
    EXPECT_EQ(executable.artifact_descriptors.front().kernel.runtime_shape_rule,
              test_case.expected_rule);
    EXPECT_EQ(executable.artifact_descriptors.front()
                  .kernel.runtime_shape_i64_metadata,
              test_case.expected_i64_metadata);

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().runtime_shape_rule,
              test_case.expected_rule);
    EXPECT_EQ(runtime_descriptor.stages.front().runtime_shape_i64_metadata,
              test_case.expected_i64_metadata);

    auto stale_descriptor = runtime_descriptor;
    stale_descriptor.stages.front().runtime_shape_rule = "stale_runtime_rule";
    const auto stale_result = compiler::verify_runtime_executable_descriptor(
        stale_descriptor, executable);
    EXPECT_FALSE(stale_result.valid());
    EXPECT_TRUE(
        has_diagnostic_containing(stale_result.diagnostics, "artifact drift"));

    if (std::string_view(test_case.expected_rule) != "static_or_descriptor") {
      auto mismatched_descriptor = runtime_descriptor;
      mismatched_descriptor.stages.front().op_family = "Relu";
      const auto mismatched_result =
          compiler::verify_runtime_executable_descriptor(mismatched_descriptor,
                                                         executable);
      EXPECT_FALSE(mismatched_result.valid());
      EXPECT_TRUE(has_diagnostic_containing(
          mismatched_result.diagnostics,
          "runtime shape rule does not match op family"));
    }

    auto stale_metadata_descriptor = runtime_descriptor;
    stale_metadata_descriptor.stages.front().runtime_shape_i64_metadata = {42};
    const auto stale_metadata_result =
        compiler::verify_runtime_executable_descriptor(
            stale_metadata_descriptor, executable);
    EXPECT_FALSE(stale_metadata_result.valid());
    EXPECT_TRUE(has_diagnostic_containing(stale_metadata_result.diagnostics,
                                          "artifact drift"));
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
    if (std::string_view(test_case.op_type) == "Concat") {
      auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 2, 3});
      auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::Shape{1, 2, 3});
      auto concat =
          std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lhs, rhs}, 1);
      op.source_node = concat;
      op.node_name = concat->get_friendly_name();
      op.type_name = concat->get_type_name();
      op.input_element_types = {"f32", "f32"};
      op.input_shapes = {"{1,2,3}", "{1,2,3}"};
      op.output_shapes = {"{1,4,3}"};
    }
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
       ModelCacheFingerprintIncludesAttributesAndConstantPayloads) {
  const auto manifest = make_single_payload_route_manifest(
      LoweringRouteKind::Metadata, "opencl", "metadata", "metadata");
  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.verify().valid());

  const auto add_one = make_add_constant_model(1.0f);
  const auto add_two = make_add_constant_model(2.0f);
  const auto add_one_fingerprint =
      compiler::make_model_cache_fingerprint(*add_one);
  const auto add_two_fingerprint =
      compiler::make_model_cache_fingerprint(*add_two);
  EXPECT_NE(add_one_fingerprint, add_two_fingerprint);

  const auto reshape_plain = make_reshape_model(false);
  const auto reshape_special_zero = make_reshape_model(true);
  EXPECT_NE(compiler::make_model_cache_fingerprint(*reshape_plain),
            compiler::make_model_cache_fingerprint(*reshape_special_zero));

  const auto envelope_one = compiler::CacheEnvelopeBuilder{}.build(
      executable, make_test_cache_options(*add_one));
  const auto envelope_two = compiler::CacheEnvelopeBuilder{}.build(
      executable, make_test_cache_options(*add_two));
  EXPECT_NE(envelope_one.key.model_fingerprint,
            envelope_two.key.model_fingerprint);
  EXPECT_NE(envelope_one.key.stable_key, envelope_two.key.stable_key);
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
  const auto adreno_module =
      compiler::make_opencl_backend_module(adreno_target);
  const auto v3d_module = compiler::make_opencl_backend_module(v3d_target);
  ASSERT_TRUE(generic_module);
  ASSERT_TRUE(adreno_module);
  ASSERT_TRUE(v3d_module);

  EXPECT_EQ(generic_module->target().fingerprint(),
            generic_target.fingerprint());
  EXPECT_EQ(adreno_module->target().fingerprint(), adreno_target.fingerprint());
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
      compiler_service.compile({make_relu_model(), target});
  ASSERT_TRUE(compile_result.supported())
      << compile_result.unsupported_message();
  EXPECT_EQ(compile_result.target.fingerprint(), target.fingerprint());
  EXPECT_EQ(compile_result.executable.target_fingerprint, target.fingerprint());

  ASSERT_TRUE(compile_result.runtime_descriptor);
  EXPECT_EQ(compile_result.runtime_descriptor->target_fingerprint,
            target.fingerprint());
  EXPECT_FALSE(compile_result.runtime_descriptor->materialization_stages.empty());
  EXPECT_EQ(
      compile_result.runtime_descriptor->runtime_options
          .custom_kernel_dispatch_profile.profile_key,
      "opencl:broadcom_v3d");
  EXPECT_EQ(compile_result.runtime_descriptor->runtime_options
                .custom_kernel_dispatch_profile
                .max_total_threads_per_group,
            64u);
  EXPECT_TRUE(compile_result.runtime_descriptor->runtime_options
                  .custom_kernel_dispatch_profile
                  .chunk_dispatch.retune_threads_to_workload);

  const auto binding_ref_compile_result =
      compiler_service.compile({make_relu_model(), target});
  ASSERT_TRUE(binding_ref_compile_result.supported())
      << binding_ref_compile_result.unsupported_message();
  ASSERT_TRUE(binding_ref_compile_result.runtime_descriptor);
  bool checked_stage_input_ref = false;
  for (const auto &stage_plan :
       binding_ref_compile_result.runtime_descriptor->materialization_stages) {
    for (const auto &input : stage_plan.io_plan.inputs) {
      if (input.source_ref.kind == PipelineStageTensorRefKind::None) {
        continue;
      }
      EXPECT_TRUE(input.source_ref.valid());
      checked_stage_input_ref = true;
    }
  }
  EXPECT_TRUE(checked_stage_input_ref);
}

TEST_F(GfxBackendArchitectureContractTest,
       BackendRegistryRequiresExactConcreteBackendTargetProfiles) {
  const auto generic_target =
      compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  compiler::StaticBackendModuleConfig config;
  config.target = generic_target;
  config.kernel_registry =
      compiler::make_common_kernel_registry(generic_target);
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

  const auto backend_module = registry.resolve(target);
  ASSERT_TRUE(backend_module);
  compiler::RuntimeExecutableDescriptorBuildRequest descriptor_request;
  descriptor_request.executable = &compile_result.executable;
  descriptor_request.transformed_model = compile_result.transformed_model;
  descriptor_request.target = target;
  descriptor_request.fusion_capabilities =
      backend_module->capabilities().fusion();

  EXPECT_THROW(
      (void)compiler::RuntimeExecutableDescriptorBuilder{}.build_finalized(
          descriptor_request),
      ov::Exception);

  descriptor_request.backend_registry = &registry;
  const auto runtime_descriptor =
      compiler::RuntimeExecutableDescriptorBuilder{}.build_finalized(
          descriptor_request);
  EXPECT_TRUE(compiler::runtime_executable_descriptor_materialization_valid(
      runtime_descriptor));
  EXPECT_EQ(
      runtime_descriptor.runtime_options.custom_kernel_dispatch_profile
          .profile_key,
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
  const auto dynamic_range_unit =
      opencl_registry.resolve_unit(LoweringRouteKind::GeneratedKernel,
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
       DescriptorOwnedRuntimeParamsStayDescriptorOnly) {
  struct Case {
    const char *backend_domain;
    KernelArtifactPayloadKind payload_kind;
  };

  const std::vector<GfxKernelBufferRole> binary_roles = {
      GfxKernelBufferRole::TensorInput,   GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::TensorOutput,  GfxKernelBufferRole::ScalarParam,
      GfxKernelBufferRole::ScalarParam,   GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams};

  for (const auto test_case :
       {Case{"metal", KernelArtifactPayloadKind::MslSource},
        Case{"opencl", KernelArtifactPayloadKind::OpenClSource}}) {
    SCOPED_TRACE(test_case.backend_domain);
    auto executable = make_source_payload_executable(
        test_case.backend_domain, "Add", binary_roles, {"{1,3}", "{1,3}"},
        {"{1,3}"});
    ASSERT_TRUE(executable.verify().valid());
    ASSERT_EQ(executable.artifact_descriptors.size(), 1u);
    EXPECT_EQ(executable.artifact_descriptors.front().payload_kind,
              test_case.payload_kind);

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().payload_kind,
              test_case.payload_kind);
    EXPECT_EQ(runtime_descriptor.stages.front().runtime_param_buffer_count, 3u);
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       SourceLaunchPlanAbiIsDescriptorOwnedAcrossSourceBackends) {
  const std::vector<GfxKernelBufferRole> roles = {
      GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::TensorOutput};

  for (const auto backend_domain : {"metal", "opencl"}) {
    SCOPED_TRACE(backend_domain);
    auto executable = make_source_payload_executable(
        backend_domain, "Add", roles, {"{1,3}", "{1,3}"}, {"{1,3}"});
    const auto executable_result = executable.verify();
    ASSERT_TRUE(executable_result.valid())
        << (executable_result.diagnostics.empty()
                ? std::string{}
                : executable_result.diagnostics.front());

    ASSERT_EQ(executable.artifact_descriptors.size(), 1u);
    const auto &artifact = executable.artifact_descriptors.front();
    ASSERT_TRUE(artifact.launch_plan.valid);
    EXPECT_EQ(artifact.launch_plan.buffer_roles,
              (std::vector<std::string>{"tensor_input", "tensor_input",
                                        "scalar_param", "tensor_output"}));
    EXPECT_EQ(artifact.launch_plan.direct_input_indices,
              (std::vector<size_t>{0, 1}));

    auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().launch_plan.buffer_roles,
              artifact.launch_plan.buffer_roles);
    EXPECT_EQ(
        runtime_descriptor.stages.front().launch_plan.direct_input_indices,
        artifact.launch_plan.direct_input_indices);

    runtime_descriptor.stages.front().launch_plan.buffer_roles.pop_back();
    const auto stale_result = compiler::verify_runtime_executable_descriptor(
        runtime_descriptor, executable);
    ASSERT_FALSE(stale_result.valid());
    EXPECT_TRUE(has_diagnostic_containing(
        stale_result.diagnostics, "source launch-plan ABI count drift"));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       SharedRuntimeParamDescriptorContractCoversGeneratedSourceFamilies) {
  struct FamilyCase {
    const char *op_family;
    size_t runtime_param_count;
    RuntimeParamDescriptorPayloadKind payload_kind;
    std::vector<GfxKernelBufferRole> roles;
    std::vector<std::string> input_shapes;
    std::vector<std::string> output_shapes;
  };

  const std::vector<FamilyCase> families = {
      {"Add",
       3,
       RuntimeParamDescriptorPayloadKind::BinaryBroadcast,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
        GfxKernelBufferRole::TensorOutput, GfxKernelBufferRole::ScalarParam,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams},
       {"{1,3}", "{1,3}"},
       {"{1,3}"}},
      {"Broadcast",
       4,
       RuntimeParamDescriptorPayloadKind::Broadcast,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams},
       {"{1,3}"},
       {"{2,3}"}},
      {"Select",
       4,
       RuntimeParamDescriptorPayloadKind::Select,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
        GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams},
       {"{1,3}", "{1,3}", "{1,3}"},
       {"{1,3}"}},
      {"Tile",
       4,
       RuntimeParamDescriptorPayloadKind::Tile,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams},
       {"{1,3}"},
       {"{2,3}"}},
  };

  for (const auto &family : families) {
    SCOPED_TRACE(family.op_family);
    EXPECT_EQ(descriptor_owned_runtime_param_payload_kind(
                  family.op_family, family.runtime_param_count),
              family.payload_kind);

    for (const auto backend_domain : {"metal", "opencl"}) {
      SCOPED_TRACE(backend_domain);
      auto executable = make_source_payload_executable(
          backend_domain, family.op_family, family.roles, family.input_shapes,
          family.output_shapes);
      ASSERT_TRUE(executable.verify().valid());

      const auto runtime_descriptor =
          compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
      ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
          runtime_descriptor, executable));
      ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
      const auto &stage = runtime_descriptor.stages.front();
      EXPECT_EQ(stage.runtime_param_buffer_count, family.runtime_param_count);
      EXPECT_TRUE(descriptor_owns_runtime_param_payload(
          stage, family.runtime_param_count));
    }
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       MslRuntimeParamDescriptorContractCoversGeneratedMetadataFamilies) {
  struct FamilyCase {
    const char *op_family;
    size_t runtime_param_count;
    RuntimeParamDescriptorPayloadKind payload_kind;
    std::vector<GfxKernelBufferRole> roles;
    std::vector<std::string> input_shapes;
    std::vector<std::string> output_shapes;
    std::vector<int64_t> i64_metadata;
    bool reduce_keep_dims = false;
    bool reduce_keep_dims_valid = false;
  };

  const std::vector<FamilyCase> families = {
      {"Softmax",
       1,
       RuntimeParamDescriptorPayloadKind::Softmax,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams},
       {"{2,3}"},
       {"{2,3}"},
       {2, 3, 1}},
      {"Transpose",
       5,
       RuntimeParamDescriptorPayloadKind::Transpose,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams},
       {"{2,3}"},
       {"{3,2}"},
       {1, 0}},
      {"ReduceSum",
       5,
       RuntimeParamDescriptorPayloadKind::Reduce,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::ScalarParam,
        GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams},
       {"{2,3}"},
       {"{2}"},
       {1},
       false,
       true},
  };

  for (const auto &family : families) {
    SCOPED_TRACE(family.op_family);
    EXPECT_EQ(descriptor_owned_runtime_param_payload_kind(
                  family.op_family, family.runtime_param_count),
              family.payload_kind);

    GfxKernelSourceRuntimeBinding runtime_binding;
    runtime_binding.runtime_param_i64_metadata = family.i64_metadata;
    runtime_binding.runtime_param_reduce_keep_dims = family.reduce_keep_dims;
    runtime_binding.runtime_param_reduce_keep_dims_valid =
        family.reduce_keep_dims_valid;

    auto executable = make_source_payload_executable(
        "metal", family.op_family, family.roles, family.input_shapes,
        family.output_shapes, runtime_binding);
    ASSERT_TRUE(executable.verify().valid());

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &stage = runtime_descriptor.stages.front();
    EXPECT_EQ(stage.runtime_param_buffer_count, family.runtime_param_count);
    EXPECT_TRUE(descriptor_owns_runtime_param_payload(
        stage, family.runtime_param_count));
    EXPECT_EQ(stage.runtime_param_i64_metadata, family.i64_metadata);
    EXPECT_EQ(stage.runtime_param_reduce_keep_dims, family.reduce_keep_dims);
    EXPECT_EQ(stage.runtime_param_reduce_keep_dims_valid,
              family.reduce_keep_dims_valid);
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeParamMetadataComesFromArtifactDescriptorNotSourcePayload) {
  const std::vector<GfxKernelBufferRole> roles = {
      GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
      GfxKernelBufferRole::RuntimeParams};

  GfxKernelSourceRuntimeBinding payload_binding;
  payload_binding.runtime_param_i64_metadata = {99, 99, 99};

  auto executable = make_source_payload_executable(
      "metal", "Softmax", roles, {"{2,3}"}, {"{2,3}"}, payload_binding);
  ASSERT_EQ(executable.artifact_descriptors.size(), 1u);
  ASSERT_EQ(executable.artifact_payloads.size(), 1u);

  auto &artifact = executable.artifact_descriptors.front();
  artifact.runtime_param_buffer_count = 1;
  artifact.runtime_param_i64_metadata = {2, 3, 1};
  artifact.runtime_param_reduce_keep_dims = false;
  artifact.runtime_param_reduce_keep_dims_valid = false;
  compiler::finalize_kernel_artifact_descriptor_identity(artifact);
  executable.artifact_payloads.front().artifact_key = artifact.artifact_key;

  const auto runtime_descriptor =
      compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
  ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(runtime_descriptor,
                                                            executable));
  ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
  const auto &stage = runtime_descriptor.stages.front();
  EXPECT_EQ(stage.runtime_param_buffer_count, 1u);
  EXPECT_EQ(stage.runtime_param_i64_metadata, (std::vector<int64_t>{2, 3, 1}));
  EXPECT_TRUE(descriptor_owns_runtime_param_payload(stage, 1));
}

TEST_F(GfxBackendArchitectureContractTest,
       ReduceRuntimeMetadataComesFromDescriptorNotSourceNode) {
  RuntimeStageExecutableDescriptor descriptor;
  descriptor.op_family = "ReduceSum";
  descriptor.stage_name = "unit_reduce_descriptor_metadata";
  descriptor.entry_point = "gfx_metal_generated_reduction_sum_f32";
  descriptor.launch_plan.valid = true;
  descriptor.launch_plan.scalar_args = {6, 2, 1};
  descriptor.runtime_param_i64_metadata = {-1};
  descriptor.runtime_param_reduce_keep_dims = false;
  descriptor.runtime_param_reduce_keep_dims_valid = true;

  const auto reduce_info = runtime_reduce_info_from_descriptor(
      descriptor, ov::Shape{2, 3}, descriptor.stage_name);

  ASSERT_TRUE(reduce_info);
  EXPECT_EQ(reduce_info->axes, ov::AxisSet{1});
  EXPECT_FALSE(reduce_info->keep_dims);

  const auto dispatch = runtime_reduce_dispatch_from_descriptor(
      descriptor, descriptor.stage_name);
  ASSERT_TRUE(dispatch.valid());
  EXPECT_EQ(dispatch.entry_point, descriptor.entry_point);
  EXPECT_EQ(dispatch.op_code, 1u);
  EXPECT_EQ(dispatch.compiler_scalar_args, (std::vector<int32_t>{6, 2, 1}));

  descriptor.launch_plan.scalar_args = {6, 2};
  EXPECT_THROW((void)runtime_reduce_dispatch_from_descriptor(
                   descriptor, descriptor.stage_name),
               ov::Exception);

  descriptor.op_family = "Softmax";
  EXPECT_FALSE(runtime_reduce_info_from_descriptor(descriptor, ov::Shape{2, 3},
                                                   descriptor.stage_name));
  EXPECT_FALSE(
      runtime_reduce_dispatch_from_descriptor(descriptor, descriptor.stage_name)
          .valid());
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeShapeRuleOwnershipIsDescriptorContract) {
  struct Case {
    const char *op_family;
    const char *runtime_shape_rule;
  };

  const std::vector<Case> positive_cases = {
      {"Concat", "concat"},         {"Broadcast", "broadcast"},
      {"Select", "select"},         {"ShapeOf", "shape_of"},
      {"Slice", "slice"},           {"StridedSlice", "slice"},
      {"Range", "range"},           {"Tile", "tile"},
      {"MatMul", "static_or_descriptor"},
  };

  for (const auto &test_case : positive_cases) {
    SCOPED_TRACE(std::string(test_case.op_family) + ":" +
                 test_case.runtime_shape_rule);
    EXPECT_TRUE(descriptor_owns_runtime_shape_rule(
        test_case.op_family, test_case.runtime_shape_rule));
  }

  const std::vector<Case> negative_cases = {
      {"Concat", "broadcast"},      {"Broadcast", "concat"},
      {"Select", "slice"},          {"ShapeOf", "range"},
      {"Slice", "tile"},            {"StridedSlice", "concat"},
      {"Range", "shape_of"},        {"Tile", "select"},
      {"ReduceSum", "reduce"},      {"Broadcast", "unknown"},
  };

  for (const auto &test_case : negative_cases) {
    SCOPED_TRACE(std::string(test_case.op_family) + ":" +
                 test_case.runtime_shape_rule);
    EXPECT_FALSE(descriptor_owns_runtime_shape_rule(
        test_case.op_family, test_case.runtime_shape_rule));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorOwnedRuntimeShapeRulesDoNotRequireSourceNodeBridge) {
  auto make_binding = [](std::string name, std::string role,
                         std::string element_type, std::string partial_shape) {
    auto binding =
        make_runtime_binding(std::move(name), "unit_region", std::move(role));
    binding.element_type = std::move(element_type);
    binding.partial_shape = std::move(partial_shape);
    return binding;
  };

  auto make_descriptor = [&](std::string backend_domain, std::string op_family,
                             std::string rule,
                             std::vector<RuntimeTensorBindingContract> inputs,
                             std::vector<RuntimeTensorBindingContract> outputs,
                             std::vector<int64_t> metadata = {}) {
    RuntimeStageExecutableDescriptor descriptor;
    descriptor.stage_index = 0;
    descriptor.stage_record_key = 0x9001u;
    descriptor.artifact_descriptor_index = 0;
    descriptor.manifest_ref = "manifest://unit/runtime_shape";
    descriptor.abi_fingerprint = "abi://unit/runtime_shape";
    descriptor.artifact_key = "artifact://unit/runtime_shape";
    descriptor.backend_domain = std::move(backend_domain);
    descriptor.kernel_id = "unit/runtime_shape";
    descriptor.op_family = std::move(op_family);
    descriptor.stage_name = descriptor.op_family + "_runtime_shape";
    descriptor.origin = KernelArtifactOrigin::Generated;
    descriptor.payload_kind = KernelArtifactPayloadKind::MslSource;
    descriptor.entry_point = "unit_runtime_shape";
    descriptor.runtime_shape_rule = std::move(rule);
    descriptor.runtime_shape_i64_metadata = std::move(metadata);
    descriptor.input_bindings = std::move(inputs);
    descriptor.output_bindings = std::move(outputs);
    return descriptor;
  };

  auto tensor = [](ov::Shape shape, ov::element::Type type) {
    GpuTensor value;
    value.shape = std::move(shape);
    value.expected_type = type;
    return value;
  };
  auto i64_tensor = [](std::vector<int64_t> values) {
    GpuTensor value;
    value.shape = ov::Shape{values.size()};
    value.expected_type = ov::element::i64;
    value.i64_values = std::move(values);
    return value;
  };

  for (const auto backend_domain : {"metal", "opencl"}) {
    SCOPED_TRACE(backend_domain);

    {
      auto descriptor = make_descriptor(
          backend_domain, "Concat", "concat",
          {make_binding("lhs", "TensorInput", "f32", "{1,2,2}"),
           make_binding("rhs", "TensorInput", "f32", "{1,3,2}")},
          {make_binding("out", "TensorOutput", "f32", "{1,5,2}")}, {1});
      std::vector<GpuTensor> storage = {tensor({1, 2, 2}, ov::element::f32),
                                        tensor({1, 3, 2}, ov::element::f32)};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan =
          plan_concat_runtime_values(inputs, descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{1, 5, 2}));
      EXPECT_EQ(plan.axis_norm, 1);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Broadcast", "broadcast",
          {make_binding("input", "TensorInput", "f32", "{1,3}"),
           make_binding("target", "TensorInput", "i64", "{2}")},
          {make_binding("out", "TensorOutput", "f32", "{2,3}")}, {0, 2});
      std::vector<GpuTensor> storage = {tensor({1, 3}, ov::element::f32),
                                        i64_tensor({2, 3})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan = plan_broadcast_runtime_values(inputs, storage[0].shape,
                                                      descriptor.stage_name);
      EXPECT_EQ(plan.output_shape, (ov::Shape{2, 3}));
      EXPECT_EQ(plan.output_type, ov::element::f32);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Select", "select",
          {make_binding("cond", "TensorInput", "boolean", "{1,3}"),
           make_binding("then", "TensorInput", "f32", "{1,3}"),
           make_binding("else", "TensorInput", "f32", "{1,3}")},
          {make_binding("out", "TensorOutput", "f32", "{1,3}")});
      std::vector<GpuTensor> storage = {tensor({1, 3}, ov::element::boolean),
                                        tensor({1, 3}, ov::element::f32),
                                        tensor({1, 3}, ov::element::f32)};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1],
                                             &storage[2]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan =
          plan_select_runtime_values(inputs, descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{1, 3}));
      EXPECT_EQ(plan.values.output_type, ov::element::f32);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "ShapeOf", "shape_of",
          {make_binding("input", "TensorInput", "f32", "{2,3,4}")},
          {make_binding("out", "TensorOutput", "i64", "{3}")});
      std::vector<GpuTensor> storage = {tensor({2, 3, 4}, ov::element::f32)};
      std::vector<GpuTensor *> input_ptrs = {&storage[0]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan =
          plan_shapeof_runtime_values(inputs, descriptor.stage_name);
      EXPECT_EQ(plan.output_shape, (ov::Shape{3}));
      EXPECT_EQ(plan.i64_values, (std::vector<int64_t>{2, 3, 4}));
    }

    {
      auto descriptor =
          make_descriptor(backend_domain, "Range", "range",
                          {make_binding("start", "TensorInput", "i64", "{1}"),
                           make_binding("stop", "TensorInput", "i64", "{1}"),
                           make_binding("step", "TensorInput", "i64", "{1}")},
                          {make_binding("out", "TensorOutput", "i64", "{3}")});
      std::vector<GpuTensor> storage = {i64_tensor({0}), i64_tensor({3}),
                                        i64_tensor({1})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1],
                                             &storage[2]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan =
          plan_range_runtime_values(inputs, descriptor.stage_name);
      EXPECT_EQ(plan.output_shape, (ov::Shape{3}));
      EXPECT_EQ(plan.i64_values, (std::vector<int64_t>{0, 1, 2}));
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Tile", "tile",
          {make_binding("input", "TensorInput", "f32", "{2,3}"),
           make_binding("repeats", "TensorInput", "i64", "{2}")},
          {make_binding("out", "TensorOutput", "f32", "{4,3}")});
      std::vector<GpuTensor> storage = {tensor({2, 3}, ov::element::f32),
                                        i64_tensor({2, 1})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      GpuTensor output;
      std::vector<GpuTensor *> outputs = {&output};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan =
          plan_tile_runtime_values(inputs, outputs, descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.output_shape, (ov::Shape{4, 3}));
      EXPECT_EQ(plan.values.output_type, ov::element::f32);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Reshape", "static_or_descriptor",
          {make_binding("input", "TensorInput", "f32", "{2,3}"),
           make_binding("pattern", "TensorInput", "i64", "{2}")},
          {make_binding("out", "TensorOutput", "f32", "{3,2}")}, {0});
      std::vector<GpuTensor> storage = {tensor({2, 3}, ov::element::f32),
                                        i64_tensor({3, 2})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan =
          plan_reshape_runtime_values(inputs, descriptor, descriptor.stage_name);
      EXPECT_EQ(plan.output_shape, (ov::Shape{3, 2}));
      EXPECT_EQ(plan.output_type, ov::element::f32);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Squeeze", "static_or_descriptor",
          {make_binding("input", "TensorInput", "f32", "{1,3,1}"),
           make_binding("axes", "TensorInput", "i64", "{2}")},
          {make_binding("out", "TensorOutput", "f32", "{3}")});
      std::vector<GpuTensor> storage = {tensor({1, 3, 1}, ov::element::f32),
                                        i64_tensor({0, 2})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan = plan_squeeze_unsqueeze_runtime_values(
          inputs, descriptor, descriptor.stage_name);
      EXPECT_EQ(plan.output_shape, (ov::Shape{3}));
      EXPECT_EQ(plan.output_type, ov::element::f32);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Unsqueeze", "static_or_descriptor",
          {make_binding("input", "TensorInput", "f32", "{3}"),
           make_binding("axes", "TensorInput", "i64", "{2}")},
          {make_binding("out", "TensorOutput", "f32", "{1,3,1}")});
      std::vector<GpuTensor> storage = {tensor({3}, ov::element::f32),
                                        i64_tensor({0, 2})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan = plan_squeeze_unsqueeze_runtime_values(
          inputs, descriptor, descriptor.stage_name);
      EXPECT_EQ(plan.output_shape, (ov::Shape{1, 3, 1}));
      EXPECT_EQ(plan.output_type, ov::element::f32);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Convert", "static_or_descriptor",
          {make_binding("input", "TensorInput", "i64", "{3}")},
          {make_binding("out", "TensorOutput", "f32", "{3}")});
      std::vector<GpuTensor> storage = {i64_tensor({1, 2, 3})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan =
          plan_convert_runtime_values(inputs, descriptor, descriptor.stage_name);
      EXPECT_EQ(plan.output_shape, (ov::Shape{3}));
      EXPECT_EQ(plan.output_type, ov::element::f32);
      EXPECT_EQ(plan.i64_values, (std::vector<int64_t>{1, 2, 3}));
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Gather", "static_or_descriptor",
          {make_binding("data", "TensorInput", "i64", "{4}"),
           make_binding("indices", "TensorInput", "i64", "{2}"),
           make_binding("axis", "TensorInput", "i64", "{1}")},
          {make_binding("out", "TensorOutput", "i64", "{2}")}, {0});
      std::vector<GpuTensor> storage = {i64_tensor({10, 20, 30, 40}),
                                        i64_tensor({1, 3}), i64_tensor({0})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1],
                                             &storage[2]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan =
          plan_gather_runtime_values(inputs, descriptor, descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{2}));
      EXPECT_EQ(plan.values.i64_values, (std::vector<int64_t>{20, 40}));
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "ScatterUpdate", "static_or_descriptor",
          {make_binding("data", "TensorInput", "f32", "{4}"),
           make_binding("indices", "TensorInput", "i64", "{2}"),
           make_binding("updates", "TensorInput", "f32", "{2}"),
           make_binding("axis", "TensorInput", "i64", "{1}")},
          {make_binding("out", "TensorOutput", "f32", "{4}")});
      std::vector<GpuTensor> storage = {tensor({4}, ov::element::f32),
                                        i64_tensor({1, 3}),
                                        tensor({2}, ov::element::f32),
                                        i64_tensor({0})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1],
                                             &storage[2], &storage[3]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan = plan_scatter_update_runtime_values(
          inputs, descriptor, descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{4}));
      EXPECT_EQ(plan.axis_norm, 0);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Split", "static_or_descriptor",
          {make_binding("input", "TensorInput", "f32", "{2,4}"),
           make_binding("axis", "TensorInput", "i64", "{1}")},
          {make_binding("out0", "TensorOutput", "f32", "{2,2}"),
           make_binding("out1", "TensorOutput", "f32", "{2,2}")});
      std::vector<GpuTensor> storage = {tensor({2, 4}, ov::element::f32),
                                        i64_tensor({1})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan =
          plan_split_runtime_values(inputs, descriptor, 2, descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.axis_norm, 1);
      EXPECT_EQ(plan.split_sizes, (std::vector<size_t>{2, 2}));
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Transpose", "static_or_descriptor",
          {make_binding("input", "TensorInput", "f32", "{2,3}"),
           make_binding("perm", "TensorInput", "i64", "{2}")},
          {make_binding("out", "TensorOutput", "f32", "{3,2}")});
      std::vector<GpuTensor> storage = {tensor({2, 3}, ov::element::f32),
                                        i64_tensor({1, 0})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1]};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan = plan_transpose_runtime_values(inputs, descriptor,
                                                      descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{3, 2}));
      EXPECT_EQ(plan.permutation, (std::vector<int64_t>{1, 0}));
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Interpolate", "static_or_descriptor",
          {make_binding("input", "TensorInput", "f32", "{1,1,2,2}")},
          {make_binding("out", "TensorOutput", "f32", "{1,1,4,4}")},
          {0, 1, 0});
      std::vector<GpuTensor> storage = {tensor({1, 1, 2, 2}, ov::element::f32)};
      std::vector<GpuTensor *> input_ptrs = {&storage[0]};
      GpuTensor output;
      std::vector<GpuTensor *> outputs = {&output};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      const auto plan = plan_interpolate_runtime_values(
          inputs, outputs, descriptor, descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{1, 1, 4, 4}));
      EXPECT_FALSE(plan.align_corners);
      EXPECT_TRUE(plan.use_half_pixel);
      EXPECT_EQ(plan.nearest_mode, 0u);
    }

    {
      auto descriptor = make_descriptor(
          backend_domain, "Slice", "slice",
          {make_binding("input", "TensorInput", "f32", "{2,5}"),
           make_binding("starts", "TensorInput", "i64", "{1}"),
           make_binding("ends", "TensorInput", "i64", "{1}"),
           make_binding("steps", "TensorInput", "i64", "{1}"),
           make_binding("axes", "TensorInput", "i64", "{1}")},
          {make_binding("out", "TensorOutput", "f32", "{2,3}")}, {1, 1, 5});
      std::vector<GpuTensor> storage = {tensor({2, 5}, ov::element::f32),
                                        i64_tensor({1}), i64_tensor({4}),
                                        i64_tensor({1}), i64_tensor({1})};
      std::vector<GpuTensor *> input_ptrs = {
          &storage[0], &storage[1], &storage[2], &storage[3], &storage[4]};
      GpuTensor output;
      std::vector<GpuTensor *> outputs = {&output};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan = plan_slice_runtime_values(inputs, outputs, false,
                                                  descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{2, 3}));
      EXPECT_EQ(plan.values.output_type, ov::element::f32);
      EXPECT_EQ(plan.starts_full, (std::vector<int32_t>{0, 1}));
      EXPECT_EQ(plan.steps_full, (std::vector<int32_t>{1, 1}));
    }

    {
      const std::vector<int64_t> strided_slice_metadata = {
          1, 2, 4, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0};
      auto descriptor = make_descriptor(
          backend_domain, "StridedSlice", "slice",
          {make_binding("input", "TensorInput", "f32", "{4,5}"),
           make_binding("begin", "TensorInput", "i64", "{2}"),
           make_binding("end", "TensorInput", "i64", "{2}"),
           make_binding("strides", "TensorInput", "i64", "{2}")},
          {make_binding("out", "TensorOutput", "f32", "{3,3}")},
          strided_slice_metadata);
      std::vector<GpuTensor> storage = {tensor({4, 5}, ov::element::f32),
                                        i64_tensor({1, 0}), i64_tensor({4, 5}),
                                        i64_tensor({1, 2})};
      std::vector<GpuTensor *> input_ptrs = {&storage[0], &storage[1],
                                             &storage[2], &storage[3]};
      GpuTensor output;
      std::vector<GpuTensor *> outputs = {&output};
      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &descriptor;
      ASSERT_EQ(inputs.descriptor, &descriptor);
      const auto plan = plan_slice_runtime_values(inputs, outputs, false,
                                                  descriptor.stage_name);
      ASSERT_TRUE(plan.valid());
      EXPECT_EQ(plan.values.output_shape, (ov::Shape{3, 3}));
      EXPECT_EQ(plan.values.output_type, ov::element::f32);
      EXPECT_EQ(plan.starts_full, (std::vector<int32_t>{1, 0}));
      EXPECT_EQ(plan.steps_full, (std::vector<int32_t>{1, 2}));
    }
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorOwnedRuntimeParamsMaterializeFromDescriptorShapes) {
  const std::vector<GfxKernelBufferRole> binary_roles = {
      GfxKernelBufferRole::TensorInput,   GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::TensorOutput,  GfxKernelBufferRole::ScalarParam,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams};

  for (const auto backend_domain : {"metal", "opencl"}) {
    SCOPED_TRACE(backend_domain);
    auto executable = make_source_payload_executable(
        backend_domain, "Add", binary_roles, {"{1,3}", "{1,3}"}, {"{1,3}"});
    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &stage = runtime_descriptor.stages.front();
    EXPECT_EQ(stage.runtime_param_buffer_count, 3u);
    ASSERT_TRUE(descriptor_owns_runtime_param_payload(stage, 3));

    UnitMetadataBufferManager buffer_manager;
    RuntimeInputResolver inputs;
    inputs.descriptor = &stage;
    GpuTensor output;
    std::vector<GpuTensor *> outputs = {&output};
    const std::vector<int32_t> compiler_scalar_args;

    auto materialization = materialize_descriptor_owned_runtime_param_payload(
        buffer_manager, stage, inputs, outputs, 3, compiler_scalar_args,
        "unit_descriptor_owned_add");

    ASSERT_TRUE(materialization.available);
    EXPECT_TRUE(materialization.descriptor_owned);
    EXPECT_EQ(materialization.extra_inputs.size(), 3u);
    EXPECT_EQ(output.shape, (ov::Shape{1, 3}));
    EXPECT_EQ(materialization.scalar_args, (std::vector<int32_t>{3, 2}));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorOwnedRuntimeParamsMaterializeDynamicBroadcastAndTileValues) {
  struct Case {
    const char *op_family;
    size_t runtime_param_count;
    std::vector<GfxKernelBufferRole> roles;
    std::vector<std::string> input_shapes;
    std::vector<std::string> output_shapes;
    ov::Shape data_shape;
    std::vector<int64_t> shape_values;
    ov::Shape expected_output_shape;
    std::vector<int32_t> expected_scalar_args;
  };

  const std::vector<Case> cases = {
      {"Broadcast",
       4,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams},
       {"{?,3}"},
       {"{?,3}"},
       ov::Shape{1, 3},
       {2, 3},
       ov::Shape{2, 3},
       {6, 2, 2}},
      {"Tile",
       4,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams},
       {"{?,3}"},
       {"{?,3}"},
       ov::Shape{2, 3},
       {2, 1},
       ov::Shape{4, 3},
       {12, 2}},
  };

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.op_family);
    for (const auto backend_domain : {"metal", "opencl"}) {
      SCOPED_TRACE(backend_domain);
      auto executable = make_source_payload_executable(
          backend_domain, test_case.op_family, test_case.roles,
          test_case.input_shapes, test_case.output_shapes);
      ASSERT_TRUE(executable.verify().valid());

      const auto runtime_descriptor =
          compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
      ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
          runtime_descriptor, executable));
      ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
      const auto &stage = runtime_descriptor.stages.front();
      ASSERT_TRUE(descriptor_owns_runtime_param_payload(
          stage, test_case.runtime_param_count));

      GpuTensor data;
      data.shape = test_case.data_shape;
      data.expected_type = ov::element::f32;
      GpuTensor shape_values;
      shape_values.shape = ov::Shape{test_case.shape_values.size()};
      shape_values.expected_type = ov::element::i64;
      shape_values.i64_values = test_case.shape_values;
      std::vector<GpuTensor *> input_ptrs = {&data, &shape_values};

      RuntimeInputResolver inputs;
      inputs.inputs = &input_ptrs;
      inputs.descriptor = &stage;

      UnitMetadataBufferManager buffer_manager;
      GpuTensor output;
      std::vector<GpuTensor *> outputs = {&output};
      const std::vector<int32_t> compiler_scalar_args;

      auto materialization = materialize_descriptor_owned_runtime_param_payload(
          buffer_manager, stage, inputs, outputs,
          test_case.runtime_param_count, compiler_scalar_args,
          test_case.op_family);

      ASSERT_TRUE(materialization.available);
      EXPECT_TRUE(materialization.descriptor_owned);
      EXPECT_EQ(materialization.extra_inputs.size(),
                test_case.runtime_param_count);
      EXPECT_EQ(output.shape, test_case.expected_output_shape);
      EXPECT_EQ(materialization.scalar_args, test_case.expected_scalar_args);
    }
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorOwnedRuntimeParamsMaterializeFromArtifactMetadata) {
  struct Case {
    const char *op_family;
    size_t runtime_param_count;
    std::vector<GfxKernelBufferRole> roles;
    std::vector<std::string> input_shapes;
    std::vector<std::string> output_shapes;
    std::vector<int64_t> i64_metadata;
    bool reduce_keep_dims = false;
    bool reduce_keep_dims_valid = false;
    std::vector<int32_t> compiler_scalar_args;
    size_t expected_extra_inputs = 0;
    ov::Shape expected_output_shape;
    std::vector<ov::Shape> runtime_input_shapes;
  };

  const std::vector<Case> cases = {
      {"Softmax",
       1,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams},
       {"{?,3}"},
       {"{?,3}"},
       {2, 3, 1},
       false,
       false,
       {},
       1,
       ov::Shape{2, 3},
       {ov::Shape{2, 3}}},
      {"Transpose",
       5,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams},
       {"{?,3}"},
       {"{3,?}"},
       {1, 0},
       false,
       false,
       {},
       5,
       ov::Shape{3, 2},
       {ov::Shape{2, 3}}},
      {"ReduceSum",
       5,
       {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
        GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::ScalarParam,
        GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
        GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams},
       {"{?,3}"},
       {"{?}"},
       {1},
       false,
       true,
       {2, 2, 0},
       5,
       ov::Shape{2},
       {ov::Shape{2, 3}}},
  };

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.op_family);
    GfxKernelSourceRuntimeBinding runtime_binding;
    runtime_binding.runtime_param_i64_metadata = test_case.i64_metadata;
    runtime_binding.runtime_param_reduce_keep_dims = test_case.reduce_keep_dims;
    runtime_binding.runtime_param_reduce_keep_dims_valid =
        test_case.reduce_keep_dims_valid;

    auto executable = make_source_payload_executable(
        "metal", test_case.op_family, test_case.roles, test_case.input_shapes,
        test_case.output_shapes, runtime_binding);
    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &stage = runtime_descriptor.stages.front();
    EXPECT_EQ(stage.runtime_param_buffer_count, test_case.runtime_param_count);
    ASSERT_TRUE(descriptor_owns_runtime_param_payload(
        stage, test_case.runtime_param_count));

    UnitMetadataBufferManager buffer_manager;
    std::vector<GpuTensor> input_storage;
    input_storage.reserve(test_case.runtime_input_shapes.size());
    for (const auto &shape : test_case.runtime_input_shapes) {
      GpuTensor input;
      input.shape = shape;
      input.expected_type = ov::element::f32;
      input_storage.push_back(std::move(input));
    }
    std::vector<GpuTensor *> input_ptrs;
    input_ptrs.reserve(input_storage.size());
    for (auto &input : input_storage) {
      input_ptrs.push_back(&input);
    }
    RuntimeInputResolver inputs;
    inputs.inputs = &input_ptrs;
    inputs.descriptor = &stage;
    GpuTensor output;
    std::vector<GpuTensor *> outputs = {&output};

    auto materialization = materialize_descriptor_owned_runtime_param_payload(
        buffer_manager, stage, inputs, outputs, test_case.runtime_param_count,
        test_case.compiler_scalar_args, test_case.op_family);

    ASSERT_TRUE(materialization.available);
    EXPECT_TRUE(materialization.descriptor_owned);
    EXPECT_EQ(materialization.extra_inputs.size(),
              test_case.expected_extra_inputs);
    EXPECT_EQ(output.shape, test_case.expected_output_shape);
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorOwnedRuntimeParamsRejectSourceNodeShapeBridge) {
  const std::vector<GfxKernelBufferRole> binary_roles = {
      GfxKernelBufferRole::TensorInput,   GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::TensorOutput,  GfxKernelBufferRole::ScalarParam,
      GfxKernelBufferRole::ScalarParam,   GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams};

  for (const auto backend_domain : {"metal", "opencl"}) {
    SCOPED_TRACE(backend_domain);
    auto executable = make_source_payload_executable(
        backend_domain, "Add", binary_roles, {"{?,3}", "{1,3}"}, {"{1,3}"});
    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &stage = runtime_descriptor.stages.front();
    ASSERT_TRUE(descriptor_owns_runtime_param_payload(stage, 3));

    UnitMetadataBufferManager buffer_manager;
    RuntimeInputResolver inputs;
    inputs.descriptor = &stage;
    GpuTensor output;
    std::vector<GpuTensor *> outputs = {&output};
    const std::vector<int32_t> compiler_scalar_args;

    EXPECT_THROW((void)materialize_descriptor_owned_runtime_param_payload(
                     buffer_manager, stage, inputs, outputs, 3,
                     compiler_scalar_args, "unit_no_source_bridge_add"),
                 ov::Exception);
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorOwnedRuntimeParamsMaterializeFromRequestTensorShapes) {
  const std::vector<GfxKernelBufferRole> binary_roles = {
      GfxKernelBufferRole::TensorInput,   GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::TensorOutput,  GfxKernelBufferRole::ScalarParam,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams};

  for (const auto backend_domain : {"metal", "opencl"}) {
    SCOPED_TRACE(backend_domain);
    auto executable = make_source_payload_executable(
        backend_domain, "Add", binary_roles, {"{?,3}", "{1,3}"}, {"{?,3}"});
    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &stage = runtime_descriptor.stages.front();
    ASSERT_TRUE(descriptor_owns_runtime_param_payload(stage, 3));

    GpuTensor lhs;
    lhs.shape = {1, 3};
    GpuTensor rhs;
    rhs.shape = {1, 3};
    std::vector<GpuTensor *> input_tensors = {&lhs, &rhs};
    RuntimeInputResolver inputs;
    inputs.inputs = &input_tensors;
    inputs.descriptor = &stage;

    UnitMetadataBufferManager buffer_manager;
    GpuTensor output;
    std::vector<GpuTensor *> outputs = {&output};
    const std::vector<int32_t> compiler_scalar_args;

    auto materialization = materialize_descriptor_owned_runtime_param_payload(
        buffer_manager, stage, inputs, outputs, 3, compiler_scalar_args,
        "unit_request_shape_owned_add");
    ASSERT_TRUE(materialization.available);
    EXPECT_TRUE(materialization.descriptor_owned);
    EXPECT_EQ(materialization.extra_inputs.size(), 3u);
    EXPECT_EQ(output.shape, (ov::Shape{1, 3}));
    EXPECT_EQ(materialization.scalar_args, (std::vector<int32_t>{3, 2}));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeParamsWithDynamicDescriptorShapePassDescriptorValidation) {
  const std::vector<GfxKernelBufferRole> binary_roles = {
      GfxKernelBufferRole::TensorInput,   GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::TensorOutput,  GfxKernelBufferRole::ScalarParam,
      GfxKernelBufferRole::ScalarParam,   GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams};

  auto executable = make_source_payload_executable(
      "opencl", "Add", binary_roles, {"{?,3}", "{1,3}"}, {"{1,3}"});
  ASSERT_TRUE(executable.verify().valid());

  const auto runtime_descriptor =
      compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
  const auto verification = compiler::verify_runtime_executable_descriptor(
      runtime_descriptor, executable);
  ASSERT_TRUE(verification.valid())
      << (verification.diagnostics.empty() ? std::string{}
                                           : verification.diagnostics.front());
  ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
  const auto &stage = runtime_descriptor.stages.front();
  EXPECT_TRUE(descriptor_owns_runtime_param_payload(stage, 3));
}

TEST_F(GfxBackendArchitectureContractTest,
       ConstTensorSourcePayloadsAreDescriptorOwned) {
  struct Case {
    const char *backend_domain;
  };

  const std::vector<GfxKernelBufferRole> const_roles = {
      GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::ConstTensor,
      GfxKernelBufferRole::TensorOutput};

  for (const auto test_case : {Case{"metal"}, Case{"opencl"}}) {
    SCOPED_TRACE(test_case.backend_domain);
    KernelArtifactConstTensor const_tensor;
    const_tensor.source_input_index = 1;
    const_tensor.logical_name = "unit_const_input";
    const_tensor.element_type = "f32";
    const_tensor.shape = {1, 3};
    const_tensor.bytes = {0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64};

    auto executable = make_source_payload_executable(
        test_case.backend_domain, "Multiply", const_roles, {"{1,3}", "{1,3}"},
        {"{1,3}"}, {}, {const_tensor});
    ASSERT_TRUE(executable.verify().valid());

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &stage = runtime_descriptor.stages.front();
    ASSERT_EQ(stage.const_tensors.size(), 1u);
    EXPECT_EQ(stage.const_tensors.front().source_input_index, 1u);
    EXPECT_EQ(stage.const_tensors.front().logical_name, "unit_const_input");
    EXPECT_EQ(stage.const_tensors.front().element_type, "f32");
    EXPECT_EQ(stage.const_tensors.front().shape, (std::vector<size_t>{1, 3}));
    EXPECT_EQ(stage.const_tensors.front().bytes, const_tensor.bytes);

    auto missing_const_executable = make_source_payload_executable(
        test_case.backend_domain, "Multiply", const_roles, {"{1,3}", "{1,3}"},
        {"{1,3}"});
    const auto missing_const_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(
            missing_const_executable);
    const auto missing_const_verification =
        compiler::verify_runtime_executable_descriptor(
            missing_const_descriptor, missing_const_executable);
    EXPECT_FALSE(missing_const_verification.valid());
    EXPECT_TRUE(has_diagnostic_containing(
        missing_const_verification.diagnostics, "ConstTensor ABI"));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorConstTensorMaterializerUsesSharedDescriptorSlots) {
  RuntimeStageExecutableDescriptor descriptor;
  descriptor.stage_name = "unit_descriptor_const_stage";
  descriptor.kernel_id = "unit/descriptor_const";
  descriptor.const_tensors = {
      KernelArtifactConstTensor{3, "rhs_const", "f32", {1}, {0, 0, 0, 64}},
      KernelArtifactConstTensor{1, "lhs_const", "f32", {1}, {0, 0, 128, 63}},
  };

  UnitDescriptorConstBufferManager buffer_manager;
  auto slots = materialize_descriptor_const_tensor_slots(
      buffer_manager, descriptor, "unit/const_tensor");

  ASSERT_EQ(slots.buffers.size(), 4u);
  ASSERT_EQ(slots.present.size(), 4u);
  EXPECT_FALSE(slots.present[0]);
  EXPECT_TRUE(slots.present[1]);
  EXPECT_FALSE(slots.present[2]);
  EXPECT_TRUE(slots.present[3]);
  EXPECT_EQ(slots.buffers[1].shape, (ov::Shape{1}));
  EXPECT_EQ(slots.buffers[3].shape, (ov::Shape{1}));
  EXPECT_EQ(slots.buffers[1].expected_type, ov::element::f32);
  EXPECT_EQ(slots.buffers[3].expected_type, ov::element::f32);
  EXPECT_TRUE(slots.buffers[1].buf.valid());
  EXPECT_TRUE(slots.buffers[3].buf.valid());

  auto args = descriptor_const_tensor_args(slots, 2);
  ASSERT_EQ(args.size(), 2u);
  EXPECT_EQ(args[0], &slots.buffers[1]);
  EXPECT_EQ(args[1], &slots.buffers[3]);
  EXPECT_NE(args[0]->buf.allocation_uid, args[1]->buf.allocation_uid);

  ASSERT_EQ(buffer_manager.uploads.size(), 2u);
  EXPECT_EQ(buffer_manager.uploads[0].bytes,
            (std::vector<uint8_t>{0, 0, 0, 64}));
  EXPECT_EQ(buffer_manager.uploads[1].bytes,
            (std::vector<uint8_t>{0, 0, 128, 63}));
  EXPECT_NE(buffer_manager.uploads[0].key, buffer_manager.uploads[1].key);
}

TEST_F(GfxBackendArchitectureContractTest,
       DescriptorConstTensorMaterializerRejectsInvalidSharedContracts) {
  RuntimeStageExecutableDescriptor duplicate_descriptor;
  duplicate_descriptor.stage_name = "unit_duplicate_const_stage";
  duplicate_descriptor.kernel_id = "unit/duplicate_const";
  duplicate_descriptor.const_tensors = {
      KernelArtifactConstTensor{1, "const_a", "f32", {1}, {0, 0, 128, 63}},
      KernelArtifactConstTensor{1, "const_b", "f32", {1}, {0, 0, 0, 64}},
  };

  UnitDescriptorConstBufferManager buffer_manager;
  EXPECT_THROW((void)materialize_descriptor_const_tensor_slots(
                   buffer_manager, duplicate_descriptor, "unit/const_tensor"),
               ov::Exception);

  RuntimeStageExecutableDescriptor cache_descriptor;
  cache_descriptor.stage_name = "unit_no_const_cache_stage";
  cache_descriptor.kernel_id = "unit/no_const_cache";
  cache_descriptor.const_tensors = {
      KernelArtifactConstTensor{0, "const_input", "f32", {1}, {0, 0, 128, 63}},
  };

  GpuBufferManager no_const_cache_manager;
  EXPECT_THROW(
      (void)materialize_descriptor_const_tensor_slots(
          no_const_cache_manager, cache_descriptor, "unit/const_tensor"),
      ov::Exception);
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
