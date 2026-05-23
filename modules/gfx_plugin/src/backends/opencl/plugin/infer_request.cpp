// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/infer_request.hpp"

#include <chrono>
#include <memory>
#include <utility>
#include <vector>

#include "backends/opencl/plugin/compiled_model_state.hpp"
#include "backends/opencl/plugin/infer_io_opencl.hpp"
#include "backends/opencl/runtime/opencl_api.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "plugin/infer_io_utils.hpp"
#include "plugin/infer_pipeline.hpp"
#include "plugin/infer_profiling_utils.hpp"
#include "plugin/infer_request_state.hpp"
#include "plugin/stateful_execution.hpp"
#include "runtime/gfx_target_profile.hpp"
#include "runtime/gpu_buffer_pool.hpp"
#include "runtime/memory_manager.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

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

OpenClInferState *get_opencl_state(InferRequestState &state) {
  return dynamic_cast<OpenClInferState *>(state.backend.get());
}

} // namespace

void OpenClBackendState::init_infer_state(InferRequestState &state) const {
  OPENVINO_ASSERT(const_manager,
                  "GFX OpenCL: buffer manager is not initialized");
  state.backend = std::make_unique<OpenClInferState>(const_manager);
}

void InferRequest::infer_opencl_impl(
    const std::shared_ptr<const CompiledModel> &cm) {
  OPENVINO_ASSERT(cm, "GFX OpenCL: compiled model is null");
  OPENVINO_ASSERT(cm->backend() == GpuBackend::OpenCL,
                  "GFX OpenCL: infer called for non-OpenCL backend");
  OPENVINO_ASSERT(cm->op_pipeline_built(),
                  "GFX OpenCL: op pipeline is not built");

  auto &state = *m_state;
  auto *opencl_state = get_opencl_state(state);
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
  void *stage_profiler = profiler ? profiler->native_handle() : nullptr;
  const auto pipeline_prepare_start =
      profiling ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
  auto &pipeline = prepare_reusable_pipeline_with_outputs(
      opencl_state->reusable_pipeline, descs, backend->const_manager.get(),
      stage_profiler, profiling, cm->get_runtime_model(), get_outputs(),
      node_map, param_map, state.bound_remote_outputs,
      state.bound_remote_inputs, GpuBackend::OpenCL, pool,
      opencl_state->stage_output_handles, &opencl_state->stage_output_workspace,
      [](std::vector<InferStage> &) {},
      [&](InferStage &stage, size_t oi, GpuTensor &out_ref, GpuBufferDesc &desc,
          const char *error_prefix) {
        const bool is_model_output =
            (oi < stage.output_is_model_output.size()) &&
            stage.output_is_model_output[oi];
        return init_stage_output_desc(GpuBackend::OpenCL, stage, oi, out_ref,
                                      desc, is_model_output,
                                      /*skip_view_ops=*/true, error_prefix);
      },
      "GFX OpenCL");
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
  prepare_reusable_execution_plan(opencl_state->reusable_execution_plan,
                                  pipeline, node_map, param_map);
  if (profiling) {
    profiler->record_segment(
        "compile", "prepare_reusable_pipeline",
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - pipeline_prepare_start));
  }

  std::vector<GpuTensor> input_tensors(get_inputs().size());
  ensure_input_handles(get_inputs().size(), /*with_staging=*/false,
                       "GFX OpenCL");
  auto &input_handles = opencl_state->input_handles;
  const auto bind_inputs_start = profiling
                                     ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
  bind_inputs_for_infer(
      GpuBackend::OpenCL,
      [&](size_t idx, const GpuTensor &dev) { input_tensors[idx] = dev; },
      [&](size_t idx, const ov::Tensor &host) {
        input_tensors[idx] = bind_host_input_opencl(
            host, &pool, &input_handles[idx], profiler, "GFX OpenCL");
      },
      "GFX OpenCL");
  if (profiling) {
    profiler->record_segment(
        "upload", "bind_inputs_for_infer",
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - bind_inputs_start));
  }

  auto input_lookup = [&](size_t input_idx) -> GpuTensor * {
    if (input_idx < input_tensors.size()) {
      return &input_tensors[input_idx];
    }
    return nullptr;
  };
  if (!opencl_state->reusable_pipeline_runtime_prewarmed) {
    const auto prewarm_start = profiling
                                   ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
    prewarm_pipeline_runtime_state(pipeline, node_map, param_map, input_lookup,
                                   &opencl_state->reusable_execution_plan);
    opencl_state->reusable_pipeline_runtime_prewarmed = true;
    if (profiling) {
      profiler->record_segment(
          "compile", "prewarm_reusable_pipeline",
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - prewarm_start));
    }
  }

  auto queue =
      reinterpret_cast<GpuCommandBufferHandle>(backend->context->queue());
  const auto infer_start = profiling ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
  execute_pipeline(
      pipeline, node_map, param_map, input_lookup,
      [&](InferStage &stage, const std::vector<GpuTensor *> &resolved_inputs) {
        try_bind_direct_stateful_assign_output(state, stage, resolved_inputs,
                                               pool, profiler);
        if (execute_stateful_stage(state, stage, resolved_inputs, pool, queue,
                                   profiler)) {
          return;
        }
        stage.stage->execute(queue);
      },
      &opencl_state->reusable_execution_plan);
  backend->context->finish();
  if (profiling) {
    profiler->record_segment(
        "infer", "execute_pipeline",
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - infer_start));
  }

  ensure_output_staging_handles(get_outputs().size(), "GFX OpenCL");
  auto &output_handles = opencl_state->output_staging_handles;
  auto command_queue =
      reinterpret_cast<GpuCommandQueueHandle>(backend->context->queue());
  const auto bind_outputs_start = profiling
                                      ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
  bind_outputs_for_infer(
      cm, pipeline, node_map, param_map, input_lookup,
      [&](size_t idx, const std::shared_ptr<GfxRemoteTensor> &remote) {
        ov::ISyncInferRequest::set_tensor(
            get_outputs()[idx], ov::SoPtr<ov::ITensor>{remote, nullptr});
      },
      [&](size_t idx, GpuTensor &dev, const OutputViewInfo &info,
          const ov::Tensor *host_override) {
        ov::Tensor *reusable_host = nullptr;
        if (idx < opencl_state->reusable_host_output_plan.outputs.size()) {
          auto &prepared = opencl_state->reusable_host_output_plan.outputs[idx];
          if (prepared.host) {
            reusable_host = &prepared.host;
          }
        }
        auto bound = bind_host_output_opencl(
            dev, info, host_override, reusable_host, &pool,
            &output_handles[idx], command_queue, profiler, "GFX OpenCL");
        ov::ISyncInferRequest::set_tensor(
            get_outputs()[idx], ov::get_tensor_impl(bound.host_tensor));
      },
      /*allow_missing=*/false, "GFX OpenCL");
  if (profiling) {
    profiler->record_segment(
        "download", "bind_outputs_for_infer",
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - bind_outputs_start));
  }

  finalize_infer_profiling("opencl", cm, state, profiler, queue);
}

} // namespace gfx_plugin
} // namespace ov
