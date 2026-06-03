// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/infer_request.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "backends/opencl/plugin/compiled_model_state.hpp"
#include "backends/opencl/plugin/infer_io_opencl.hpp"
#include "backends/opencl/runtime/opencl_api.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "plugin/infer_request_backend_access.hpp"
#include "plugin/infer_io_utils.hpp"
#include "plugin/infer_profiling_utils.hpp"
#include "plugin/infer_request_state.hpp"
#include "runtime/gfx_target_profile.hpp"
#include "runtime/gpu_buffer_pool.hpp"
#include "runtime/infer_executor.hpp"
#include "runtime/infer_pipeline.hpp"
#include "runtime/infer_submission.hpp"
#include "runtime/memory_manager.hpp"
#include "runtime/stateful_execution.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

constexpr size_t opencl_mib(size_t value) {
  return value * 1024u * 1024u;
}

constexpr uint64_t opencl_gmacs(uint64_t value) {
  return value * 1000ull * 1000ull * 1000ull;
}

void apply_opencl_submission_profile(const GpuExecutionDeviceInfo &info,
                                     InferSubmissionTuningCaps &caps) {
  const uint32_t simd =
      std::max<uint32_t>(1u,
                         std::max(info.subgroup_size,
                                  info.preferred_simd_width));
  const uint32_t max_threads =
      std::max<uint32_t>(info.max_total_threads_per_group, 1u);

  if (max_threads > 256u) {
    caps.max_slots_hint = 6u;
    caps.stages_per_slot_hint = simd >= 64u ? 72u : 56u;
  } else if (max_threads > 128u) {
    caps.extremely_deep_source_stage_floor =
        (simd >= 64u ? 64u : 48u) + (simd >= 64u ? 32u : 24u);
    caps.extremely_deep_source_output_floor =
        simd >= 64u ? opencl_mib(96u) : opencl_mib(64u);
    caps.extremely_deep_source_mac_floor =
        simd >= 64u ? opencl_gmacs(8u) : opencl_gmacs(4u);
    caps.extremely_deep_dependency_extension_budget_num = 5u;
    caps.extremely_deep_dependency_extension_budget_den = 4u;
  } else {
    caps.extremely_deep_dependency_extension_budget_num = 5u;
    caps.extremely_deep_dependency_extension_budget_den = 4u;
  }

  if (info.device_family == GpuDeviceFamily::BroadcomV3D) {
    caps.mac_budget_scale_num = simd >= 64u ? 3u : 2u;
    caps.mac_budget_scale_den = simd >= 64u ? 4u : 3u;
  }
}

class OpenClBufferManagerAllocator final : public IGpuAllocator {
public:
  explicit OpenClBufferManagerAllocator(OpenClBufferManager &manager)
      : m_manager(manager) {}

  GpuBackend backend() const override { return GpuBackend::OpenCL; }

  GpuBuffer allocate(const GpuBufferDesc &desc) override {
    return m_manager.allocate_temp(desc);
  }

  GpuBuffer wrap_shared(void *, size_t, ov::element::Type) override {
    OPENVINO_THROW("GFX OpenCL: host shared memory import is not implemented");
  }

  void release(GpuBuffer &&buf) override {
    m_manager.release_temp(std::move(buf));
  }

private:
  OpenClBufferManager &m_manager;
};

void release_opencl_buffer_handles(
    std::vector<BufferHandle> &handles,
    const std::shared_ptr<OpenClBufferManager> &manager) {
  if (!manager) {
    return;
  }
  for (auto &handle : handles) {
    if (handle.valid()) {
      manager->release_temp(std::move(handle.buf));
    }
    handle.capacity = 0;
  }
}

struct OpenClInferState final : BackendInferState {
  explicit OpenClInferState(std::shared_ptr<OpenClBufferManager> manager)
      : buffer_manager(std::move(manager)) {}

  ~OpenClInferState() override {
    release_opencl_buffer_handles(input_handles, buffer_manager);
    release_opencl_buffer_handles(input_staging_handles, buffer_manager);
    for (auto &handles : stage_output_handles) {
      release_opencl_buffer_handles(handles, buffer_manager);
    }
    release_opencl_buffer_handles(stage_output_workspace.handles,
                                  buffer_manager);
    release_opencl_buffer_handles(output_staging_handles, buffer_manager);
  }

  std::shared_ptr<OpenClBufferManager> buffer_manager;
};

OpenClInferState *get_opencl_state(BackendRequestState &state) {
  return dynamic_cast<OpenClInferState *>(state.backend.get());
}

class OpenClInferSubmissionSession final
    : public SingleFlightInferSubmissionSession {
public:
  OpenClInferSubmissionSession(OpenClRuntimeContext &context,
                               GfxProfiler *profiler)
      : m_context(context), m_profiler(profiler) {}

  bool supports_incremental_submit() const override { return false; }

protected:
  GpuCommandBufferHandle begin_recording_on_slot() override {
    auto *queue = m_context.queue();
    OPENVINO_ASSERT(queue, "GFX OpenCL: command queue is null");
    return reinterpret_cast<GpuCommandBufferHandle>(queue);
  }

  void submit_recorded_on_slot(GpuCommandBufferHandle /*command_buffer*/,
                               bool /*continue_recording*/) override {}

  void finish_submission_slot() override {
    const bool profiling = (m_profiler != nullptr);
    const auto finish_start = profiling ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
    m_context.finish();
    auto queue = m_context.queue();
    set_completed_command_buffer(
        reinterpret_cast<GpuCommandBufferHandle>(queue));
    if (profiling) {
      m_profiler->record_segment(
          "wait", "opencl_finish",
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - finish_start));
      m_profiler->increment_counter("opencl_finish_count");
    }
  }

private:
  OpenClRuntimeContext &m_context;
  GfxProfiler *m_profiler = nullptr;
};

} // namespace

void OpenClBackendState::init_infer_state(BackendRequestState &state) const {
  OPENVINO_ASSERT(const_manager,
                  "GFX OpenCL: buffer manager is not initialized");
  state.backend = std::make_unique<OpenClInferState>(const_manager);
}

void execute_opencl_infer_request(
    InferRequest& request,
    const std::shared_ptr<const CompiledModel> &cm) {
  OPENVINO_ASSERT(cm, "GFX OpenCL: compiled model is null");
  OPENVINO_ASSERT(cm->backend() == GpuBackend::OpenCL,
                  "GFX OpenCL: infer called for non-OpenCL backend");
  OPENVINO_ASSERT(cm->op_pipeline_built(),
                  "GFX OpenCL: op pipeline is not built");

  auto &state = InferRequestBackendAccess::state(request);
  auto *opencl_state = get_opencl_state(state.runtime);
  OPENVINO_ASSERT(opencl_state,
                  "GFX OpenCL: infer backend state is not initialized");
  auto *backend = dynamic_cast<const OpenClBackendState *>(cm->backend_state());
  OPENVINO_ASSERT(backend && backend->context && backend->const_manager,
                  "GFX OpenCL: backend runtime is not initialized");
  opencl_state->buffer_manager = backend->const_manager;

  OpenClBufferManagerAllocator allocator(*backend->const_manager);
  GpuBufferPool pool(allocator);

  GfxProfiler *profiler = prepare_infer_profiler(*cm, state, "GFX OpenCL");
  const bool profiling = (profiler != nullptr);
  if (profiler) {
    profiler->begin_infer(cm->pipeline_desc().size());
    if (auto info = backend->const_manager->query_execution_device_info()) {
      record_gfx_target_profile(make_gfx_target_profile(*info), profiler);
    }
  }

  const auto &descs = cm->pipeline_desc();
  const auto &node_map = cm->node_to_stage();
  const auto &param_map = cm->parameter_index();
  const auto runtime_model = cm->get_runtime_model();
  void *stage_profiler = profiler ? profiler->native_handle() : nullptr;

  std::vector<GpuTensor> input_tensors;
  InferRequestBackendAccess::bind_inputs_before_infer(
      request,
      GpuBackend::OpenCL,
      input_tensors,
      [&](size_t /*idx*/, const ov::Tensor &host, BufferHandle *device_handle) {
        return bind_host_input_opencl(host, &pool, device_handle, profiler,
                                      "GFX OpenCL");
      },
      {},
      profiler,
      profiling,
      /*with_staging=*/false,
      "GFX OpenCL");

  auto input_lookup = [&](size_t input_idx) -> GpuTensor * {
    return lookup_runtime_input_tensor(input_tensors, input_idx);
  };
  OpenClInferSubmissionSession submission(*backend->context, profiler);
  InferSubmissionTuningCaps submission_caps{};
  if (const auto info = backend->const_manager->query_execution_device_info()) {
    submission_caps.preferred_simd_width =
        std::max<uint32_t>(info->preferred_simd_width, 1u);
    submission_caps.subgroup_size = std::max<uint32_t>(info->subgroup_size, 1u);
    submission_caps.max_total_threads_per_group =
        std::max<uint32_t>(info->max_total_threads_per_group, 1u);
    apply_opencl_submission_profile(*info, submission_caps);
  }
  InferRuntimeExecutionConfig execution_config{};
  execution_config.state = opencl_state;
  execution_config.descs = &descs;
  execution_config.buffer_manager = backend->const_manager.get();
  execution_config.stage_profiler = stage_profiler;
  execution_config.profiling_enabled = profiling;
  execution_config.runtime_model = &runtime_model;
  execution_config.public_outputs = &InferRequestBackendAccess::outputs(request);
  execution_config.node_map = &node_map;
  execution_config.param_map = &param_map;
  execution_config.remote_outputs = &state.bound_remote_outputs;
  execution_config.remote_inputs = &state.bound_remote_inputs;
  execution_config.expected_backend = GpuBackend::OpenCL;
  execution_config.runtime_descriptor = cm->runtime_descriptor();
  execution_config.pool = &pool;
  execution_config.post_prepare = [](std::vector<InferStage> &) {};
  execution_config.runtime_input_tensors = &input_tensors;
  execution_config.init_output_desc =
      [&](InferStage &stage, size_t oi, GpuTensor &out_ref,
          GpuBufferDesc &desc, const char *error_prefix) {
        const bool is_model_output =
            (oi < stage.output_is_model_output.size()) &&
            stage.output_is_model_output[oi];
        return init_stage_output_desc(GpuBackend::OpenCL, stage, oi, out_ref,
                                      desc, is_model_output,
                                      /*skip_view_ops=*/true, error_prefix);
      };
  execution_config.input_lookup = input_lookup;
  execution_config.submission = &submission;
  execution_config.submission_caps = submission_caps;
  execution_config.on_stage =
      [&](InferStage &stage, const std::vector<GpuTensor *> &resolved_inputs,
          GpuCommandBufferHandle command_buffer) {
        execute_infer_stage_with_stateful_contract(
            state.variable_states, stage, resolved_inputs, pool, command_buffer, profiler);
      };
  execution_config.profiler = profiler;
  execution_config.error_prefix = "GFX OpenCL";
  const auto execution_result =
      prepare_and_execute_infer_runtime(std::move(execution_config));
  auto &pipeline = *execution_result.pipeline;
  auto completed_command_buffer = execution_result.completed_command_buffer;
  if (profiler) {
    profiler->set_counter(
        "stage_output_workspace_outputs",
        opencl_state->stage_output_workspace.last_workspace_outputs);
    profiler->set_counter(
        "stage_output_direct_outputs",
        opencl_state->stage_output_workspace.last_direct_outputs);
    profiler->set_counter("stage_output_workspace_slots",
                          opencl_state->stage_output_workspace.last_slots_used);
    profiler->set_counter(
        "stage_output_workspace_peak_live",
        opencl_state->stage_output_workspace.last_peak_live_slots);
  }

  auto command_queue =
      reinterpret_cast<GpuCommandQueueHandle>(backend->context->queue());
  InferRequestBackendAccess::bind_outputs_after_infer(
      request,
      cm,
      pipeline,
      input_lookup,
      [&](size_t /*idx*/, GpuTensor &dev, const OutputViewInfo &info,
          const ov::Tensor *host_override, ov::Tensor *reusable_host,
          BufferHandle *staging_handle) {
        return bind_host_output_opencl(dev, info, host_override, reusable_host,
                                       &pool, staging_handle, command_queue,
                                       profiler, "GFX OpenCL");
      },
      {},
      profiler,
      profiling,
      "GFX OpenCL");

  finalize_infer_profiling("opencl", cm, state, profiler,
                           completed_command_buffer);
}

} // namespace gfx_plugin
} // namespace ov
