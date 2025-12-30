// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/infer_request.hpp"

#include <iostream>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <sstream>

#import <Metal/Metal.h>
#ifdef NO
#undef NO
#endif
#ifdef YES
#undef YES
#endif

#include "openvino/gfx_plugin/compiled_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "plugin/infer_request_backend_hooks.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gpu_buffer_pool.hpp"
#include "runtime/memory_manager.hpp"
#include "backends/metal/plugin/compiled_model_state.hpp"
#include "backends/metal/runtime/gpu_memory.hpp"
#include "backends/metal/runtime/metal_memory.hpp"
#include "backends/metal/runtime/profiling/profiler.hpp"
#include "backends/metal/plugin/infer_io_metal.hpp"
#include "plugin/infer_request_state.hpp"
#include "infer_pipeline.hpp"
#include "infer_io_utils.hpp"

namespace ov {
namespace gfx_plugin {

struct MetalInferState {
    MetalAllocatorCore* alloc_core = nullptr;
    MetalDeviceCaps caps{};
    MetalTensorMap tensor_map;
    MetalHeapPool heaps;
    MetalFreeList freelist;
    MetalStagingPool staging;
    MetalAllocator allocator;

    MetalInferState(MetalAllocatorCore& core, const MetalDeviceCaps& caps_in)
        : alloc_core(&core),
          caps(caps_in),
          heaps(core),
          freelist(),
          staging(core),
          allocator(core, heaps, freelist, staging, caps_in) {}
};

void MetalInferStateDeleter::operator()(MetalInferState* ptr) const {
    delete ptr;
}

void init_backend_infer_state(InferRequestState& state, const CompiledModel& cm) {
    if (cm.backend() != GpuBackend::Metal) {
        return;
    }
    auto* metal = dynamic_cast<const MetalBackendState*>(cm.backend_state());
    OPENVINO_ASSERT(metal, "GFX: Metal backend state is not initialized");
    OPENVINO_ASSERT(metal->alloc_core, "GFX: Metal allocator core is not initialized");
    state.metal.reset(new MetalInferState(*metal->alloc_core, metal->caps));
}

ov::SoPtr<ov::ITensor> get_backend_tensor_override(const InferRequestState& state,
                                                   size_t idx,
                                                   const std::vector<ov::Output<const ov::Node>>& outputs) {
    if (!state.metal || !state.metal->tensor_map.has_output_device(idx)) {
        return {};
    }
    const auto& dev = state.metal->tensor_map.get_output_device(idx);
    if (dev.buf.storage_mode != static_cast<uint32_t>(MTLStorageModeShared)) {
        OPENVINO_THROW("GFX: output buffer is not host-visible");
    }
    id<MTLBuffer> buf = static_cast<id<MTLBuffer>>(dev.buf.buffer);
    void* ptr = buf ? [buf contents] : nullptr;
    if (!ptr) {
        OPENVINO_THROW("GFX: shared output buffer has no CPU pointer");
    }
    ov::Shape shape = dev.shape;
    if (shape.empty()) {
        const auto& out = outputs.at(idx);
        if (out.get_partial_shape().is_static())
            shape = out.get_shape();
        else
            shape = ov::Shape{1};
    }
    ov::element::Type logical = dev.expected_type == ov::element::dynamic ? dev.buf.type : dev.expected_type;
    ov::Tensor view{logical, shape, ptr};
    return ov::get_tensor_impl(view);
}

void InferRequest::infer_metal_impl(const std::shared_ptr<const CompiledModel>& cm) {
    @autoreleasepool {
        OPENVINO_ASSERT(cm, "CompiledModel is null");
        OPENVINO_ASSERT(cm->backend() == GpuBackend::Metal, "GFX: Metal infer called for non-metal backend");

        auto& state = *m_state;
        OPENVINO_ASSERT(state.metal && state.metal->alloc_core, "MetalAllocator is not initialized");
        auto* alloc_core = state.metal->alloc_core;
        auto& allocator = state.metal->allocator;
        auto& tensor_map = state.metal->tensor_map;
        auto& caps = state.metal->caps;

        if (!state.debug_buffers.empty()) {
            for (auto& buf : state.debug_buffers) {
                allocator.release(std::move(buf));
            }
            state.debug_buffers.clear();
            state.debug_tensors.clear();
        }

        tensor_map.reset_inference(alloc_core);
        allocator.reset_stats();

        MetalMemorySession session(allocator, nullptr, nullptr);
        MetalBufferManager::set_current_session(&session);
        struct SessionGuard {
            MetalMemorySession& session;
            ~SessionGuard() {
                session.end();
                MetalBufferManager::set_current_session(nullptr);
            }
        } session_guard{session};

        for_each_input_tensor(
            get_inputs().size(),
            state.bound_remote_inputs,
            [&](size_t idx) {
                return resolve_remote_input_tensor(idx, GpuBackend::Metal, "GFX");
            },
            [&](size_t idx) {
                return resolve_host_input_tensor(idx);
            },
            [&](size_t idx, const GpuTensor& dev) {
                tensor_map.bind_input_device(idx, dev);
            },
            [&](size_t idx, const ov::Tensor& host) {
                ov::Tensor src = host;
                if (!src) {
                    ov::Shape sh = get_inputs()[idx].get_partial_shape().is_static()
                                       ? get_inputs()[idx].get_shape()
                                       : ov::Shape{1};
                    src = ov::Tensor{get_inputs()[idx].get_element_type(), sh};
                }
                // Bind host -> device using zero-copy shared buffer (no memcpy).
                const auto dev = bind_host_input_metal(src,
                                                       alloc_core,
                                                       "GFX");
                tensor_map.bind_input_device(idx, dev);
            });

    MetalProfiler* profiler = nullptr;
    if (cm && cm->enable_profiling()) {
        state.profiler_cfg.level = cm->profiling_level();
        const bool detailed = (state.profiler_cfg.level == ProfilingLevel::Detailed);
        state.profiler_cfg.include_segments = detailed;
        state.profiler_cfg.include_allocations = detailed;
        state.profiler_cfg.include_transfers = detailed;
        if (state.profiler_cfg.level != ProfilingLevel::Off) {
            if (!state.metal_profiler) {
                auto* metal = dynamic_cast<const MetalBackendState*>(cm->backend_state());
                OPENVINO_ASSERT(metal, "GFX: Metal backend state is not initialized");
                state.metal_profiler = std::make_unique<MetalProfiler>(state.profiler_cfg, caps, metal->device);
            } else {
                state.metal_profiler->set_config(state.profiler_cfg);
            }
            profiler = state.metal_profiler.get();
        }
    }
    const bool detailed = (state.profiler_cfg.level == ProfilingLevel::Detailed);
    allocator.set_profiler(profiler, detailed);

    std::vector<InferStage> pipeline;
    auto run_pipeline = [&]() {
        auto shape_to_string = [](const ov::Shape& shape) {
            std::ostringstream ss;
            ss << "[";
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i) ss << ",";
                ss << shape[i];
            }
            ss << "]";
            return ss.str();
        };
        auto tensor_type = [](const MetalTensor& t) {
            return t.expected_type == ov::element::dynamic ? t.buf.type : t.expected_type;
        };
        auto tensor_bytes = [&](const MetalTensor& t, const ov::Shape& shape) -> size_t {
            if (shape.empty())
                return 0;
            const auto ty = tensor_type(t);
            return ov::shape_size(shape) * ty.size();
        };
        auto safe_check = [&](const char* label,
                              const MetalTensor* t,
                              const ov::Shape& shape,
                              bool require_valid) {
            if (!t || !t->buf.valid()) {
                if (require_valid) {
                    OPENVINO_ASSERT(false, "GFX: ", label, " buffer is null");
                }
                return;
            }
            if (shape.empty())
                return;
            const size_t need = tensor_bytes(*t, shape);
            OPENVINO_ASSERT(need <= t->buf.size,
                            "GFX: ", label, " buffer overflow (need ", need,
                            ", have ", t->buf.size, ", shape=", shape_to_string(shape), ")");
        };

        OPENVINO_ASSERT(cm->op_pipeline_built() && cm->op_pipeline_size() > 0,
                        "GFX: op pipeline is not built");
        const auto& descs = cm->pipeline_desc();
        const auto& node_map = cm->node_to_stage();
        const auto& param_map = cm->parameter_index();

        const bool profiling_enabled = (profiler != nullptr);
        auto* metal = dynamic_cast<const MetalBackendState*>(cm->backend_state());
        OPENVINO_ASSERT(metal && metal->const_manager, "GFX: Metal buffer manager is not initialized");
        pipeline = build_bound_pipeline(descs,
                                        metal->const_manager.get(),
                                        profiler,
                                        profiling_enabled,
                                        get_outputs(),
                                        node_map,
                                        param_map,
                                        state.bound_remote_outputs,
                                        state.bound_remote_inputs,
                                        GpuBackend::Metal,
                                        "GFX");

        // Allocate outputs (reuse buffers across iterations via handles).
        MetalGpuAllocator gpu_alloc(allocator, *alloc_core, caps);
        GpuBufferPool pool(gpu_alloc);
        allocate_stage_outputs(
            pipeline,
            state.stage_output_handles,
            pool,
            [&](InferStage& stage,
                size_t oi,
                GpuTensor& out_ref,
                GpuBufferDesc& desc,
                const char* error_prefix) {
                const bool is_model_output = (oi < stage.output_is_model_output.size()) &&
                                             stage.output_is_model_output[oi];
                return init_stage_output_desc(GpuBackend::Metal,
                                              stage,
                                              oi,
                                              out_ref,
                                              desc,
                                              is_model_output,
                                              /*skip_view_ops=*/false,
                                              error_prefix);
            },
            "GFX");

        if (profiler) {
            const size_t expected_samples = pipeline.size() * 4;
            profiler->begin_infer(expected_samples);
        }

        // Create a single command buffer for the entire pipeline.
        OPENVINO_ASSERT(metal && metal->command_queue, "GFX: command queue is null");
        id<MTLCommandQueue> cq = static_cast<id<MTLCommandQueue>>(metal->command_queue);
        OPENVINO_ASSERT(cq, "GFX: command queue is null");
        id<MTLCommandBuffer> cb = [cq commandBuffer];

        for (const auto& stage : pipeline) {
            auto resolved = resolve_stage_inputs(stage,
                                                 node_map,
                                                 param_map,
                                                 pipeline,
                                                 [&](size_t input_idx) -> GpuTensor* {
                                                     return &tensor_map.get_input_device(input_idx);
                                                 });
            if (gfx_log_debug_enabled() || metal_safe_debug_enabled()) {
                const std::string node_name =
                    stage.node ? stage.node->get_friendly_name() : stage.stage->name();
                const std::string node_type =
                    stage.node ? stage.node->get_type_name() : stage.stage->type();
                if (gfx_log_debug_enabled()) {
                    std::ostringstream oss;
                    oss << "Op=" << node_type << " name=" << node_name;
                    for (size_t i = 0; i < resolved.size(); ++i) {
                        if (!resolved[i])
                            continue;
                        const auto& t = *resolved[i];
                        oss << " in" << i << "_shape=" << shape_to_string(t.shape)
                            << " in" << i << "_bytes=" << t.buf.size;
                    }
                    for (size_t i = 0; i < stage.outputs.size(); ++i) {
                        const auto& t = stage.outputs[i];
                        if (!t)
                            continue;
                        oss << " out" << i << "_shape=" << shape_to_string(t->shape)
                            << " out" << i << "_bytes=" << t->buf.size;
                    }
                    GFX_LOG_DEBUG("PIPELINE", oss.str());
                }
                for (size_t i = 0; i < resolved.size(); ++i) {
                    if (!resolved[i])
                        continue;
                    ov::Shape in_shape = resolved[i]->shape;
                    if (in_shape.empty() && stage.node &&
                        stage.node->get_input_partial_shape(i).is_static()) {
                        in_shape = stage.node->get_input_shape(i);
                    }
                    safe_check(("input" + std::to_string(i)).c_str(), resolved[i], in_shape, true);
                }
                for (size_t i = 0; i < stage.outputs.size(); ++i) {
                    if (!stage.outputs[i])
                        continue;
                    ov::Shape out_shape = stage.outputs[i]->shape;
                    if (out_shape.empty() && stage.node &&
                        stage.node->get_output_partial_shape(i).is_static()) {
                        out_shape = stage.node->get_output_shape(i);
                    }
                    safe_check(("output" + std::to_string(i)).c_str(),
                               stage.outputs[i].get(),
                               out_shape,
                               false);
                }
            }
            stage.stage->set_inputs(resolved);
            stage.stage->execute(cb);
        }

        [cb commit];
        [cb waitUntilCompleted];

        state.last_profiling.clear();
        if (profiler) {
            const auto stats = allocator.stats();
            profiler->set_memory_stats(stats);
            profiler->end_infer(cb);
            state.last_profiling = profiler->export_ov();
            if (cm) {
                cm->update_last_profiling_report_json(profiler->export_extended().to_json());
            }
        } else if (cm) {
            MetalProfilingReport empty;
            empty.level = ProfilingLevel::Off;
            cm->update_last_profiling_report_json(empty.to_json());
        }
    };

    run_pipeline();

    auto output_input_lookup = [&](size_t input_idx) -> GpuTensor* {
        if (!tensor_map.has_input_device(input_idx)) {
            return nullptr;
        }
        return &tensor_map.get_input_device(input_idx);
    };
    auto* metal = dynamic_cast<const MetalBackendState*>(cm->backend_state());
    OPENVINO_ASSERT(metal && metal->command_queue, "GFX: command queue is null");
    bind_outputs_common(
        get_outputs(),
        cm->get_runtime_model(),
        cm->node_to_stage(),
        cm->parameter_index(),
        pipeline,
        output_input_lookup,
        state.bound_remote_outputs,
        [&](size_t idx, const ov::element::Type& type, const ov::Shape& shape, const char* error_prefix) {
            return get_host_output_override(idx, type, shape, error_prefix);
        },
        [&](size_t idx, const std::shared_ptr<GfxRemoteTensor>& remote) {
            ov::ISyncInferRequest::set_tensor(get_outputs()[idx],
                                              ov::SoPtr<ov::ITensor>{remote, nullptr});
        },
        [&](size_t idx,
            GpuTensor& dev,
            const OutputViewInfo& info,
            const ov::Tensor* host_override) {
            auto bound = bind_host_output_metal(dev,
                                                info,
                                                host_override,
                                                alloc_core,
                                                &allocator,
                                                metal->command_queue,
                                                "GFX");
            tensor_map.bind_output_device(idx, bound.device_tensor);
            ov::ISyncInferRequest::set_tensor(get_outputs()[idx], ov::get_tensor_impl(bound.host_tensor));
        },
        /*allow_missing=*/false,
        /*allow_fallback_one=*/true,
        "GFX");

    if (const char* mem_flag = std::getenv("OV_GFX_MEM_STATS")) {
        (void)mem_flag;
        auto stats = allocator.stats();
        double mb = 1024.0 * 1024.0;
        std::ostringstream oss;
        oss << "[GFX][mem] H2D=" << (stats.h2d_bytes / mb) << "MB "
            << "D2H=" << (stats.d2h_bytes / mb) << "MB "
            << "alloc=" << (stats.bytes_allocated_total / mb) << "MB "
            << "reuse_hits=" << stats.num_reuse_hits;
        std::cerr << oss.str() << std::endl;
    }

    if (cm) {
        if (auto* metal = dynamic_cast<const MetalBackendState*>(cm->backend_state())) {
            metal->last_stats = allocator.stats();
        }
    }
    }
}

}  // namespace gfx_plugin
}  // namespace ov
