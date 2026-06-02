// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/mps_graph_attention_stage.hpp"

#include <chrono>
#include <sstream>
#include <utility>
#include <vector>

#include "backends/metal/runtime/mpsrt/mpsrt_context.hpp"
#include "backends/metal/runtime/mpsrt/mpsrt_request.hpp"
#include "openvino/core/except.hpp"
#include "backends/metal/runtime/mpsrt/gfx_mpsrt_model.hpp"
#include "backends/metal/runtime/mpsrt/gfx_mpsrt_plan.hpp"
#include "runtime/gfx_profiler.hpp"
#include "runtime/gpu_buffer_manager.hpp"
#include "runtime/gpu_tensor.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

namespace runtime_mpsrt = ::ov::gfx_plugin::mpsrt;

std::vector<int64_t> shape_to_i64(const ov::Shape& shape) {
  std::vector<int64_t> dims;
  dims.reserve(shape.size());
  for (auto dim : shape) {
    dims.push_back(static_cast<int64_t>(dim));
  }
  return dims;
}

GfxMpsrtTensorDesc make_external_ndarray_desc(const ov::Shape& shape,
                                              const ov::element::Type& type) {
  return gfx_mpsrt_make_tensor_desc(shape_to_i64(shape), type,
                                    GfxStageStorageKind::NDArray,
                                    GfxMpsrtTensorFlagExternalIo);
}

class MpsGraphAttentionStage final : public GpuStage {
public:
  MpsGraphAttentionStage(VendorAttentionStageSpec spec,
                         MetalDeviceHandle device,
                         MetalCommandQueueHandle queue)
      : m_spec(std::move(spec)), m_device(device), m_queue(queue) {
    m_name = m_spec.name.empty() ? "mps_graph_attention" : m_spec.name;
  }

  void init(GpuBufferManager* buffer_manager) override {
    m_buffer_manager = buffer_manager;
  }

  void compile(GpuBufferManager* buffer_manager) override {
    if (buffer_manager) {
      m_buffer_manager = buffer_manager;
    }
    OPENVINO_ASSERT(m_device, "GFX MPSGraphAttention: Metal device handle is null");
    OPENVINO_ASSERT(m_spec.element_type == ov::element::f32 ||
                        m_spec.element_type == ov::element::f16,
                    "GFX MPSGraphAttention: only f16/f32 is supported");
    OPENVINO_ASSERT(m_spec.query_shape.size() == 4 &&
                        m_spec.key_shape.size() == 4 &&
                        m_spec.value_shape.size() == 4 &&
                        m_spec.output_shape.size() == 4,
                    "GFX MPSGraphAttention: Q/K/V/output must be rank-4");

    const auto query_desc =
        make_external_ndarray_desc(m_spec.query_shape, m_spec.element_type);
    const auto key_desc =
        make_external_ndarray_desc(m_spec.key_shape, m_spec.element_type);
    const auto value_desc =
        make_external_ndarray_desc(m_spec.value_shape, m_spec.element_type);
    const auto output_desc =
        make_external_ndarray_desc(m_spec.output_shape, m_spec.element_type);

    GfxMpsrtSdpaAbiDesc sdpa_desc{};
    sdpa_desc.scale = m_spec.scale;
    sdpa_desc.accumulate_fp32 = 1;
    sdpa_desc.layout = GfxMpsrtSdpaLayoutTransposedBHDN;

    GfxMpsrtStageDesc stage_desc{};
    stage_desc.kind = GfxMpsrtStageKind::MPSSdpa;
    stage_desc.domain = GfxStageBackendDomain::AppleMps;
    stage_desc.input_storage = GfxMpsrtStorage::NDArray;
    stage_desc.output_storage = GfxMpsrtStorage::NDArray;
    stage_desc.layout = GfxMpsrtLayout::RowMajor;
    stage_desc.kernel_name = "mps_sdpa";
    stage_desc.stage_manifest = make_gfx_vendor_stage_manifest(
        GfxKernelStageFamily::AttentionSoftmax, GfxKernelBackendDomain::AppleMps,
        GfxKernelStorageKind::NDArray, "apple_mps:ndarray:TransposedSDPA");
    stage_desc.sdpa_desc = sdpa_desc;

    runtime_mpsrt::MpsrtModel model;
    model.stage_record_key = gfx_mpsrt_stage_record_key(stage_desc);
    model.semantic_input_values = {0, 1, 2};
    model.semantic_output_values = {3};
    model.input_values = {0, 1, 2};
    model.output_values = {3};
    model.external_values = {0, 1, 2, 3};
    model.external_input_values = {0, 1, 2};
    model.external_output_values = {3};
    model.external_buffer_roles = {GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorInput,
                                   GfxMpsrtExternalBufferRole::TensorOutput};
    model.tensors.push_back({0, gfx_mpsrt_to_abi_desc(query_desc)});
    model.tensors.push_back({1, gfx_mpsrt_to_abi_desc(key_desc)});
    model.tensors.push_back({2, gfx_mpsrt_to_abi_desc(value_desc)});
    model.tensors.push_back({3, gfx_mpsrt_to_abi_desc(output_desc)});

    runtime_mpsrt::MpsrtRuntimeStage stage;
    stage.kind = GfxMpsrtStageKind::MPSSdpa;
    stage.stage_record_key = model.stage_record_key;
    stage.kernel_name = "mps_sdpa";
    stage.sdpa_desc = sdpa_desc;
    stage.inputs = {0, 1, 2};
    stage.outputs = {3};
    stage.output_descs = {gfx_mpsrt_to_abi_desc(output_desc)};
    model.stages.push_back(stage);

    std::string error;
    OPENVINO_ASSERT(runtime_mpsrt::finalize_mpsrt_model_resources(model, &error),
                    error);
    auto context =
        std::make_shared<metal::mpsrt::MpsrtContext>((id<MTLDevice>)m_device);
    auto prepared = std::make_shared<metal::mpsrt::MpsrtPreparedModel>();
    OPENVINO_ASSERT(context->prepare_model(model, "", *prepared, &error), error);

    m_model = std::move(model);
    m_context = std::move(context);
    m_prepared = std::move(prepared);
  }

  void execute(GpuCommandBufferHandle command_buffer) override {
    OPENVINO_ASSERT(m_prepared, "GFX MPSGraphAttention: stage is not compiled");
    OPENVINO_ASSERT(m_inputs.size() == 3 && m_inputs[0] && m_inputs[1] &&
                        m_inputs[2],
                    "GFX MPSGraphAttention: Q/K/V inputs are not bound");
    std::vector<GpuTensor*> outputs = m_outputs;
    if (outputs.empty() && m_output) {
      outputs.push_back(m_output);
    }
    OPENVINO_ASSERT(outputs.size() == 1 && outputs[0],
                    "GFX MPSGraphAttention: output is not bound");

    auto* output = outputs.front();
    if (output->shape.empty()) {
      output->shape = m_spec.output_shape;
    }
    if (output->expected_type == ov::element::dynamic) {
      output->expected_type = m_spec.element_type;
    }

    metal::mpsrt::MpsrtRequestBindingSet binding_set;
    binding_set.bindings.bind(
        0, metal::mpsrt::MpsrtBoundBuffer{m_inputs[0]->buf.buffer,
                                          m_inputs[0]->buf.offset});
    binding_set.bindings.bind(
        1, metal::mpsrt::MpsrtBoundBuffer{m_inputs[1]->buf.buffer,
                                          m_inputs[1]->buf.offset});
    binding_set.bindings.bind(
        2, metal::mpsrt::MpsrtBoundBuffer{m_inputs[2]->buf.buffer,
                                          m_inputs[2]->buf.offset});
    binding_set.bindings.bind(
        3, metal::mpsrt::MpsrtBoundBuffer{output->buf.buffer,
                                          output->buf.offset});

    KernelExecutionHooks hooks;
    KernelExecutionHooks* hooks_ptr = nullptr;
    auto* profiler = static_cast<GfxProfiler*>(m_profiler);
    if (m_profiling_enabled && profiler) {
      hooks.stage_name = m_name;
      hooks.stage_type = m_type;
      hooks.on_counter = [profiler](std::string_view name, uint64_t delta) {
        profiler->increment_counter(name, delta);
      };
      hooks.on_segment =
          [profiler](std::string_view phase, std::string_view name,
                     std::chrono::microseconds cpu_us, uint64_t gpu_us,
                     uint32_t dispatches, uint64_t bytes_in,
                     uint64_t bytes_out, uint64_t macs_est,
                     uint64_t flops_est, int64_t inflight_slot,
                     uint64_t queue_id, uint64_t cmd_buffer_id) {
            profiler->record_segment(phase, name, cpu_us, gpu_us, dispatches,
                                     bytes_in, bytes_out, macs_est, flops_est,
                                     inflight_slot, queue_id, cmd_buffer_id);
          };
      hooks_ptr = &hooks;
    }

    std::vector<KernelDispatch> dispatches(m_model.stages.size());
    metal::mpsrt::MpsrtRequest request;
    metal::mpsrt::MpsrtModelEncodeResult result;
    std::string error;
    const bool encoded = request.encode_prepared_model_with_binding_set(
        command_buffer, *m_context, m_model, *m_prepared, dispatches, binding_set,
        hooks_ptr, &result, &error);
    OPENVINO_ASSERT(encoded, error);
  }

  void set_inputs(const std::vector<GpuTensor*>& inputs) override {
    m_inputs = inputs;
  }

  void set_output(GpuTensor* output) override {
    m_output = output;
    m_outputs.clear();
    if (output) {
      m_outputs.push_back(output);
    }
  }

  void set_output_refs(const std::vector<GpuTensor*>& outputs) override {
    m_outputs = outputs;
    m_output = outputs.empty() ? nullptr : outputs.front();
  }

  void set_outputs(const std::vector<std::unique_ptr<GpuTensor>>& outputs) override {
    m_outputs.clear();
    m_outputs.reserve(outputs.size());
    for (const auto& output : outputs) {
      m_outputs.push_back(output.get());
    }
    m_output = m_outputs.empty() ? nullptr : m_outputs.front();
  }

  void enable_profiling(bool enable) override {
    m_profiling_enabled = enable;
  }

  void set_profiler(void* profiler, uint32_t node_id,
                    const std::string& node_name,
                    const std::string& node_type) override {
    m_profiler = profiler;
    m_profile_node_id = node_id;
    m_profile_node_name = node_name;
    m_profile_node_type = node_type;
  }

  const std::string& name() const override { return m_name; }
  const std::string& type() const override { return m_type; }

  std::unique_ptr<GpuStage> clone() const override {
    auto stage = std::make_unique<MpsGraphAttentionStage>(m_spec, m_device, m_queue);
    stage->m_model = m_model;
    stage->m_context = m_context;
    stage->m_prepared = m_prepared;
    stage->m_profiling_enabled = m_profiling_enabled;
    stage->m_profiler = m_profiler;
    stage->m_profile_node_id = m_profile_node_id;
    stage->m_profile_node_name = m_profile_node_name;
    stage->m_profile_node_type = m_profile_node_type;
    return stage;
  }

private:
  VendorAttentionStageSpec m_spec;
  MetalDeviceHandle m_device = nullptr;
  [[maybe_unused]] MetalCommandQueueHandle m_queue = nullptr;
  GpuBufferManager* m_buffer_manager = nullptr;
  std::string m_name;
  std::string m_type = "MpsGraphAttention";
  std::vector<GpuTensor*> m_inputs;
  std::vector<GpuTensor*> m_outputs;
  GpuTensor* m_output = nullptr;
  runtime_mpsrt::MpsrtModel m_model;
  std::shared_ptr<metal::mpsrt::MpsrtContext> m_context;
  std::shared_ptr<metal::mpsrt::MpsrtPreparedModel> m_prepared;
  bool m_profiling_enabled = false;
  void* m_profiler = nullptr;
  uint32_t m_profile_node_id = 0;
  std::string m_profile_node_name;
  std::string m_profile_node_type;
};

}  // namespace

std::unique_ptr<GpuStage> create_metal_vendor_attention_stage(
    VendorAttentionStageSpec spec,
    MetalDeviceHandle device,
    MetalCommandQueueHandle queue) {
  return std::make_unique<MpsGraphAttentionStage>(std::move(spec), device, queue);
}

}  // namespace gfx_plugin
}  // namespace ov
