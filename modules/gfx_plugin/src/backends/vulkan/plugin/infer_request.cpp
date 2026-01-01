// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/infer_request.hpp"

#include <cstring>
#include <string>

#include "openvino/gfx_plugin/compiled_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "runtime/memory_manager.hpp"
#include "runtime/gpu_buffer_pool.hpp"
#include "runtime/gpu_types.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/gfx_profiler.hpp"
#include "backends/vulkan/runtime/gpu_memory.hpp"
#include "backends/vulkan/runtime/vulkan_backend.hpp"
#include "backends/vulkan/runtime/vulkan_memory.hpp"
#include "backends/vulkan/plugin/infer_io_vulkan.hpp"
#include "plugin/gfx_profiling_utils.hpp"
#include "plugin/infer_profiling_utils.hpp"
#include "plugin/infer_request_state.hpp"
#include "runtime/gfx_remote_context.hpp"
#include "infer_pipeline.hpp"
#include "infer_io_utils.hpp"
#include "backends/vulkan/plugin/compiled_model_state.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

struct VulkanInferState final : BackendInferState {
    std::vector<BufferHandle> input_handles;
    std::vector<BufferHandle> input_staging_handles;
    std::vector<BufferHandle> output_staging_handles;

    ~VulkanInferState() override {
        VulkanGpuAllocator allocator;
        for (auto& handle : input_handles) {
            if (handle.valid()) {
                allocator.release(std::move(handle.buf));
            }
        }
        for (auto& handle : input_staging_handles) {
            if (handle.valid()) {
                allocator.release(std::move(handle.buf));
            }
        }
        for (auto& stage_handles : stage_output_handles) {
            for (auto& handle : stage_handles) {
                if (handle.valid()) {
                    allocator.release(std::move(handle.buf));
                }
            }
        }
        for (auto& handle : output_staging_handles) {
            if (handle.valid()) {
                allocator.release(std::move(handle.buf));
            }
        }
    }
};

VulkanInferState* get_vulkan_state(InferRequestState& state) {
    return dynamic_cast<VulkanInferState*>(state.backend.get());
}

const VulkanInferState* get_vulkan_state(const InferRequestState& state) {
    return dynamic_cast<const VulkanInferState*>(state.backend.get());
}

}  // namespace

void VulkanBackendState::init_infer_state(InferRequestState& state) const {
    state.backend = std::make_unique<VulkanInferState>();
}

void InferRequest::infer_vulkan_impl(const std::shared_ptr<const CompiledModel>& cm) {
    OPENVINO_ASSERT(cm, "CompiledModel is null");
    if (cm->backend() != GpuBackend::Vulkan) {
        OPENVINO_THROW("GFX backend '",
                       backend_to_string(cm->backend()),
                       "' is not supported by Vulkan infer path");
    }
    auto& state = *m_state;
    auto* vk_state = get_vulkan_state(state);
    OPENVINO_ASSERT(vk_state, "GFX: Vulkan infer state is not initialized");
    OPENVINO_ASSERT(cm->op_pipeline_built() && cm->op_pipeline_size() > 0,
                    "GFX: op pipeline is not built");

    const auto& descs = cm->pipeline_desc();
    const auto& node_map = cm->node_to_stage();
    const auto& param_map = cm->parameter_index();

    GfxProfiler* profiler = prepare_infer_profiler(*cm, state, "GFX Vulkan");
    if (profiler) {
        profiler->begin_infer(descs.size());
    }
    const bool profiling_enabled = (profiler != nullptr);
    void* stage_profiler = profiler ? profiler->native_handle() : nullptr;
    VulkanGpuAllocator allocator;
    GpuBufferPool pool(allocator);
    auto pipeline = build_pipeline_with_outputs(
        descs,
        nullptr,
        stage_profiler,
        profiling_enabled,
        get_outputs(),
        node_map,
        param_map,
        state.bound_remote_outputs,
        state.bound_remote_inputs,
        GpuBackend::Vulkan,
        pool,
        vk_state->stage_output_handles,
        [&](std::vector<InferStage>& stages) {
            for (auto& stage : stages) {
                for (auto& out : stage.outputs) {
                    if (out) {
                        out->prefer_private = true;
                    }
                }
            }
        },
        [&](InferStage& stage,
            size_t oi,
            GpuTensor& out_ref,
            GpuBufferDesc& desc,
            const char* error_prefix) {
            const bool is_model_output =
                (oi < stage.output_is_model_output.size()) && stage.output_is_model_output[oi];
            return init_stage_output_desc(GpuBackend::Vulkan,
                                          stage,
                                          oi,
                                          out_ref,
                                          desc,
                                          is_model_output,
                                          /*skip_view_ops=*/true,
                                          error_prefix);
        },
        "GFX Vulkan");

    // Create device buffers for inputs (or use remote buffers when provided).
    std::vector<GpuTensor> input_tensors(get_inputs().size());
    if (vk_state->input_handles.size() < get_inputs().size()) {
        vk_state->input_handles.resize(get_inputs().size());
    }
    if (vk_state->input_staging_handles.size() < get_inputs().size()) {
        vk_state->input_staging_handles.resize(get_inputs().size());
    }
    for_each_input_tensor(
        get_inputs().size(),
        state.bound_remote_inputs,
        [&](size_t idx) {
            return resolve_remote_input_tensor(idx, GpuBackend::Vulkan, "GFX Vulkan");
        },
        [&](size_t idx) {
            return resolve_host_input_tensor(idx);
        },
        [&](size_t idx, const GpuTensor& dev) {
            input_tensors[idx] = dev;
        },
        [&](size_t idx, const ov::Tensor& host) {
            input_tensors[idx] = bind_host_input_vulkan(host,
                                                        &pool,
                                                        &vk_state->input_handles[idx],
                                                        &vk_state->input_staging_handles[idx],
                                                        "GFX Vulkan");
        });

    // Execute pipeline.
    execute_pipeline(
        pipeline,
        node_map,
        param_map,
        [&](size_t input_idx) -> GpuTensor* {
            if (input_idx < input_tensors.size()) {
                return &input_tensors[input_idx];
            }
            return nullptr;
        },
        [&](InferStage& stage, const std::vector<GpuTensor*>& /*resolved*/) {
            stage.stage->execute(nullptr);
        });

    finalize_infer_profiling("vulkan",
                             cm,
                             state,
                             profiler,
                             nullptr);

    // Copy outputs back to host tensors via host-visible staging buffers.
    const auto runtime_model = cm->get_runtime_model();
    const auto& public_outputs = get_outputs();
    if (vk_state->output_staging_handles.size() < get_outputs().size()) {
        vk_state->output_staging_handles.resize(get_outputs().size());
    }
    auto output_input_lookup = [&](size_t input_idx) -> GpuTensor* {
        if (input_idx < input_tensors.size()) {
            return &input_tensors[input_idx];
        }
        return nullptr;
    };
    bind_outputs_common(
        public_outputs,
        runtime_model,
        node_map,
        param_map,
        pipeline,
        output_input_lookup,
        state.bound_remote_outputs,
        [&](size_t idx, const ov::element::Type& type, const ov::Shape& shape, const char* error_prefix) {
            return get_host_output_override(idx, type, shape, error_prefix);
        },
        [&](size_t idx, const std::shared_ptr<GfxRemoteTensor>& remote) {
            ov::ISyncInferRequest::set_tensor(get_outputs().at(idx),
                                              ov::SoPtr<ov::ITensor>{remote, nullptr});
        },
        [&](size_t idx, GpuTensor& dev, const OutputViewInfo& info, const ov::Tensor* host_override) {
            auto bound = bind_host_output_vulkan(dev,
                                                 info,
                                                 host_override,
                                                 &pool,
                                                 &vk_state->output_staging_handles[idx],
                                                 "GFX Vulkan");
            ov::ISyncInferRequest::set_tensor(get_outputs().at(idx),
                                              ov::get_tensor_impl(bound.host_tensor));
        },
        /*allow_missing=*/false,
        /*allow_fallback_one=*/false,
        "GFX Vulkan");

}

}  // namespace gfx_plugin
}  // namespace ov
