// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request.hpp"

#include <cstring>

#include "compiled_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "runtime/gpu_memory.hpp"
#include "runtime/gpu_buffer_pool.hpp"
#include "runtime/gpu_types.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "backends/vulkan/runtime/gpu_memory.hpp"
#include "backends/vulkan/profiling/profiler.hpp"
#include "backends/vulkan/runtime/backend.hpp"
#include "backends/vulkan/runtime/memory.hpp"
#include "remote_stub.hpp"
#include "infer_pipeline.hpp"
#include "infer_io_utils.hpp"

namespace ov {
namespace gfx_plugin {

void InferRequest::infer_vulkan_impl(const std::shared_ptr<const CompiledModel>& cm) {
    OPENVINO_ASSERT(cm, "CompiledModel is null");
    if (cm->backend() != GpuBackend::Vulkan) {
        OPENVINO_THROW("GFX backend '",
                       backend_to_string(cm->backend()),
                       "' is not supported by Vulkan infer path");
    }
    OPENVINO_ASSERT(cm->op_pipeline_built() && cm->op_pipeline_size() > 0,
                    "GFX: op pipeline is not built");

    const auto& descs = cm->pipeline_desc();
    const auto& node_map = cm->node_to_stage();
    const auto& param_map = cm->parameter_index();

    VulkanProfiler* profiler = nullptr;
    if (cm->enable_profiling()) {
        m_profiler_cfg.level = cm->profiling_level();
        if (m_profiler_cfg.level != ProfilingLevel::Off) {
            if (!m_vulkan_profiler) {
                auto& ctx = VulkanContext::instance();
                m_vulkan_profiler = std::make_unique<VulkanProfiler>(ctx.device(),
                                                                     ctx.physical_device(),
                                                                     ctx.queue_family_index());
            }
            m_vulkan_profiler->set_config(m_profiler_cfg);
            profiler = m_vulkan_profiler.get();
            profiler->begin_infer(descs.size());
        }
    }
    const bool profiling_enabled = (profiler != nullptr);
    auto pipeline = build_bound_pipeline(descs,
                                         nullptr,
                                         profiler,
                                         profiling_enabled,
                                         get_outputs(),
                                         node_map,
                                         param_map,
                                         m_bound_remote_outputs,
                                         m_bound_remote_inputs,
                                         GpuBackend::Vulkan,
                                         "GFX Vulkan");
    for (auto& stage : pipeline) {
        for (auto& out : stage.outputs) {
            if (out) {
                out->prefer_private = true;
            }
        }
    }

    // Create device buffers for inputs (or use remote buffers when provided).
    VulkanGpuAllocator allocator;
    GpuBufferPool pool(allocator);
    std::vector<GpuTensor> input_tensors(get_inputs().size());
    if (m_vulkan_input_handles.size() < get_inputs().size()) {
        m_vulkan_input_handles.resize(get_inputs().size());
    }
    if (m_vulkan_input_staging_handles.size() < get_inputs().size()) {
        m_vulkan_input_staging_handles.resize(get_inputs().size());
    }
    for_each_input_tensor(
        get_inputs().size(),
        m_bound_remote_inputs,
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
            input_tensors[idx] = bind_host_input(host,
                                                 GpuBackend::Vulkan,
                                                 nullptr,
                                                 &pool,
                                                 &m_vulkan_input_handles[idx],
                                                 &m_vulkan_input_staging_handles[idx],
                                                 "GFX Vulkan");
        });

    // Allocate output buffers (reuse shapes where known).
    prepare_stage_output_handles(m_vulkan_stage_output_handles,
                                 pipeline,
                                 pool,
                                 /*release_view_only=*/true);
    allocate_stage_outputs(
        pipeline,
        m_vulkan_stage_output_handles,
        pool,
        [&](InferStage& stage,
            size_t oi,
            GpuTensor& out_ref,
            GpuBufferDesc& desc,
            const char* error_prefix) {
            return init_stage_output_desc(GpuBackend::Vulkan,
                                          stage,
                                          oi,
                                          out_ref,
                                          desc,
                                          /*is_model_output=*/false,
                                          /*skip_view_ops=*/true,
                                          error_prefix);
        },
        "GFX Vulkan");

    // Execute pipeline.
    for (auto& stage : pipeline) {
        auto resolved = resolve_stage_inputs(stage,
                                             node_map,
                                             param_map,
                                             pipeline,
                                             [&](size_t input_idx) -> GpuTensor* {
                                                 if (input_idx < input_tensors.size()) {
                                                     return &input_tensors[input_idx];
                                                 }
                                                 return nullptr;
                                             });
        stage.stage->set_inputs(resolved);
        stage.stage->execute(nullptr);
    }

    m_last_profiling.clear();
    if (profiler) {
        profiler->end_infer();
        m_last_profiling = profiler->export_ov();
    }

    // Copy outputs back to host tensors via host-visible staging buffers.
    const auto runtime_model = cm->get_runtime_model();
    const auto& public_outputs = get_outputs();
    if (m_vulkan_output_staging_handles.size() < get_outputs().size()) {
        m_vulkan_output_staging_handles.resize(get_outputs().size());
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
        m_bound_remote_outputs,
        [&](size_t idx, const ov::element::Type& type, const ov::Shape& shape, const char* error_prefix) {
            return get_host_output_override(idx, type, shape, error_prefix);
        },
        [&](size_t idx, const std::shared_ptr<GfxRemoteTensor>& remote) {
            ov::ISyncInferRequest::set_tensor(get_outputs().at(idx),
                                              ov::SoPtr<ov::ITensor>{remote, nullptr});
        },
        [&](size_t idx, GpuTensor& dev, const OutputViewInfo& info, const ov::Tensor* host_override) {
            auto bound = bind_host_output(dev,
                                          info,
                                          host_override,
                                          GpuBackend::Vulkan,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          &pool,
                                          &m_vulkan_output_staging_handles[idx],
                                          "GFX Vulkan");
            ov::ISyncInferRequest::set_tensor(get_outputs().at(idx),
                                              ov::get_tensor_impl(bound.host_tensor));
        },
        /*allow_missing=*/false,
        /*allow_fallback_one=*/false,
        "GFX Vulkan");

}

void InferRequest::release_vulkan_cache() {
    VulkanGpuAllocator allocator;
    for (auto& handle : m_vulkan_input_handles) {
        if (handle.valid()) {
            allocator.release(std::move(handle.buf));
        }
        handle.capacity = 0;
    }
    m_vulkan_input_handles.clear();
    for (auto& handle : m_vulkan_input_staging_handles) {
        if (handle.valid()) {
            allocator.release(std::move(handle.buf));
        }
        handle.capacity = 0;
    }
    m_vulkan_input_staging_handles.clear();
    for (auto& stage_handles : m_vulkan_stage_output_handles) {
        for (auto& handle : stage_handles) {
            if (handle.valid()) {
                allocator.release(std::move(handle.buf));
            }
            handle.capacity = 0;
        }
    }
    m_vulkan_stage_output_handles.clear();
    for (auto& handle : m_vulkan_output_staging_handles) {
        if (handle.valid()) {
            allocator.release(std::move(handle.buf));
        }
        handle.capacity = 0;
    }
    m_vulkan_output_staging_handles.clear();
}

}  // namespace gfx_plugin
}  // namespace ov
