// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request.hpp"

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

#include "compiled_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/core/shape_util.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/gpu_buffer_pool.hpp"
#include "runtime/gpu_memory.hpp"
#include "backends/metal/runtime/gpu_memory.hpp"
#include "backends/metal/runtime/memory.hpp"
#include "backends/metal/profiling/profiler.hpp"
#include "backends/vulkan/profiling/profiler.hpp"
#include "infer_pipeline.hpp"
#include "infer_io_utils.hpp"
#include "remote_stub.hpp"

namespace ov {
namespace gfx_plugin {

InferRequest::InferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    // Allocate host tensors for inputs/outputs to satisfy AsyncInferRequest checks.
    for (const auto& input : get_inputs()) {
        allocate_tensor(input, [input](ov::SoPtr<ov::ITensor>& tensor) {
            tensor = ov::make_tensor(input.get_element_type(),
                                     input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape());
        });
    }
    for (const auto& output : get_outputs()) {
        allocate_tensor(output, [output](ov::SoPtr<ov::ITensor>& tensor) {
            tensor = ov::make_tensor(output.get_element_type(),
                                     output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape());
        });
    }
    m_bound_inputs.resize(get_inputs().size());
    m_bound_remote_inputs.resize(get_inputs().size());
    m_bound_output_hosts.resize(get_outputs().size());
    m_bound_remote_outputs.resize(get_outputs().size());
    if (auto cm = get_compiled_model_typed()) {
        if (cm->backend() != GpuBackend::Metal) {
            return;
        }
        m_alloc_core = &cm->allocator_core();
        m_caps = cm->device_caps();
        m_heaps = std::make_unique<MetalHeapPool>(*m_alloc_core);
        m_freelist = std::make_unique<MetalFreeList>();
        m_staging = std::make_unique<MetalStagingPool>(*m_alloc_core);
        m_allocator = std::make_unique<MetalAllocator>(*m_alloc_core, *m_heaps, *m_freelist, *m_staging, m_caps);
    }
}

InferRequest::~InferRequest() {
    release_vulkan_cache();
}

void InferRequest::set_input_tensor(const ov::Tensor& tensor) {
    // Single-input convenience: index 0
    set_input_tensor(0, tensor);
}

void InferRequest::set_input_tensor(size_t idx, const ov::Tensor& tensor) {
    if (idx >= m_bound_inputs.size())
        m_bound_inputs.resize(get_inputs().size());

    if (idx >= m_bound_remote_inputs.size())
        m_bound_remote_inputs.resize(get_inputs().size());

    auto impl = ov::get_tensor_impl(tensor);
    if (std::dynamic_pointer_cast<ov::IRemoteTensor>(impl._ptr)) {
        auto remote = std::dynamic_pointer_cast<GfxRemoteTensor>(impl._ptr);
        OPENVINO_ASSERT(remote, "GFX: remote tensor type mismatch");
        const auto cm = get_compiled_model_typed();
        const auto backend = cm ? cm->backend() : GpuBackend::Metal;
        const char* backend_name = backend_to_string(backend);
        OPENVINO_ASSERT(remote->backend() == backend,
                        "GFX: remote tensor backend mismatch (expected ", backend_name, ")");
        ov::ISyncInferRequest::set_tensor(get_inputs().at(idx), impl);
        m_bound_inputs[idx] = {};
        m_bound_remote_inputs[idx] = remote;
        return;
    }

    // Cache a host view (no CPU copy)
    m_bound_inputs[idx] = tensor;
    m_bound_remote_inputs[idx] = {};
    ov::ISyncInferRequest::set_tensor(get_inputs().at(idx), impl);
}

void InferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    auto remote = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr);
    const auto& outputs = get_outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i] == port) {
            break;
        }
    }
    if (remote) {
        auto gfx_remote = std::dynamic_pointer_cast<GfxRemoteTensor>(remote);
        OPENVINO_ASSERT(gfx_remote, "GFX: remote tensor type mismatch");
        const auto cm = get_compiled_model_typed();
        const auto backend = cm ? cm->backend() : GpuBackend::Metal;
        const char* backend_name = backend_to_string(backend);
        OPENVINO_ASSERT(gfx_remote->backend() == backend,
                        "GFX: remote tensor backend mismatch (expected ", backend_name, ")");
        ov::ISyncInferRequest::set_tensor(port, tensor);
    } else {
        ov::ISyncInferRequest::set_tensor(port, tensor);
    }

    auto& inputs = get_inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i] == port) {
            if (i >= m_bound_inputs.size())
                m_bound_inputs.resize(inputs.size());
            if (i >= m_bound_remote_inputs.size())
                m_bound_remote_inputs.resize(inputs.size());
            if (remote) {
                m_bound_inputs[i] = {};
                m_bound_remote_inputs[i] = std::dynamic_pointer_cast<GfxRemoteTensor>(remote);
            } else {
                ov::Tensor stored_view = ov::make_tensor(ov::ISyncInferRequest::get_tensor(port));
                m_bound_inputs[i] = stored_view;
                m_bound_remote_inputs[i] = {};
            }
            break;
        }
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i] == port) {
            if (i >= m_bound_remote_outputs.size())
                m_bound_remote_outputs.resize(outputs.size());
            if (remote) {
                m_bound_remote_outputs[i] = std::dynamic_pointer_cast<GfxRemoteTensor>(remote);
            } else {
                m_bound_remote_outputs[i] = {};
                if (i >= m_bound_output_hosts.size())
                    m_bound_output_hosts.resize(outputs.size());
                m_bound_output_hosts[i] = ov::make_tensor(tensor);
            }
            break;
        }
    }
}

ov::SoPtr<ov::ITensor> InferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    auto found = find_port(port);
    if (found.found() && found.is_output()) {
        size_t idx = found.idx;
        if (idx < m_bound_remote_outputs.size() && m_bound_remote_outputs[idx]) {
            return ov::SoPtr<ov::ITensor>{m_bound_remote_outputs[idx], nullptr};
        }
        if (m_tensor_map.has_output_device(idx)) {
            const auto& dev = m_tensor_map.get_output_device(idx);
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
                const auto& out = get_outputs().at(idx);
                if (out.get_partial_shape().is_static())
                    shape = out.get_shape();
                else
                    shape = ov::Shape{1};
            }
            ov::element::Type logical = dev.expected_type == ov::element::dynamic ? dev.buf.type : dev.expected_type;
            ov::Tensor view{logical, shape, ptr};
            return ov::get_tensor_impl(view);
        }
    }
    return ov::ISyncInferRequest::get_tensor(port);
}

ov::Tensor InferRequest::get_output_tensor(size_t idx) const {
    auto so = get_tensor(get_outputs().at(idx));
    return ov::make_tensor(so);
}

void InferRequest::infer() {
    @autoreleasepool {
        auto cm = get_compiled_model_typed();
        OPENVINO_ASSERT(cm, "CompiledModel is null");
        if (cm->backend() != GpuBackend::Metal) {
            infer_vulkan_impl(cm);
            return;
        }

        OPENVINO_ASSERT(m_allocator && m_alloc_core, "MetalAllocator is not initialized");

        if (!m_debug_buffers.empty() && m_allocator) {
            for (auto& buf : m_debug_buffers) {
                m_allocator->release(std::move(buf));
            }
            m_debug_buffers.clear();
            m_debug_tensors.clear();
        }

        m_tensor_map.reset_inference(m_alloc_core);
        m_allocator->reset_stats();

        MetalMemorySession session(*m_allocator, nullptr, nullptr);
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
            m_bound_remote_inputs,
            [&](size_t idx) {
                return resolve_remote_input_tensor(idx, GpuBackend::Metal, "GFX");
            },
            [&](size_t idx) {
                return resolve_host_input_tensor(idx);
            },
            [&](size_t idx, const GpuTensor& dev) {
                m_tensor_map.bind_input_device(idx, dev);
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
                const auto dev = bind_host_input(src,
                                                 GpuBackend::Metal,
                                                 m_alloc_core,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 "GFX");
                m_tensor_map.bind_input_device(idx, dev);
            });

    MetalProfiler* profiler = nullptr;
    if (cm && cm->enable_profiling()) {
        m_profiler_cfg.level = cm->profiling_level();
        const bool detailed = (m_profiler_cfg.level == ProfilingLevel::Detailed);
        m_profiler_cfg.include_segments = detailed;
        m_profiler_cfg.include_allocations = detailed;
        m_profiler_cfg.include_transfers = detailed;
        if (m_profiler_cfg.level != ProfilingLevel::Off) {
            if (!m_profiler) {
                m_profiler = std::make_unique<MetalProfiler>(m_profiler_cfg, m_caps, cm->device_handle());
            } else {
                m_profiler->set_config(m_profiler_cfg);
            }
            profiler = m_profiler.get();
        }
    }
    if (m_allocator) {
        const bool detailed = (m_profiler_cfg.level == ProfilingLevel::Detailed);
        m_allocator->set_profiler(profiler, detailed);
    }

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
        pipeline = build_bound_pipeline(descs,
                                        cm->const_manager().get(),
                                        profiler,
                                        profiling_enabled,
                                        get_outputs(),
                                        node_map,
                                        param_map,
                                        m_bound_remote_outputs,
                                        m_bound_remote_inputs,
                                        GpuBackend::Metal,
                                        "GFX");

        // Allocate outputs (reuse buffers across iterations via handles).
        MetalGpuAllocator gpu_alloc(*m_allocator, *m_alloc_core, m_caps);
        GpuBufferPool pool(gpu_alloc);
    allocate_stage_outputs(
        pipeline,
        m_stage_output_handles,
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
        id<MTLCommandQueue> cq = static_cast<id<MTLCommandQueue>>(cm->command_queue());
        OPENVINO_ASSERT(cq, "GFX: command queue is null");
        id<MTLCommandBuffer> cb = [cq commandBuffer];

        for (const auto& stage : pipeline) {
            auto resolved = resolve_stage_inputs(stage,
                                                 node_map,
                                                 param_map,
                                                 pipeline,
                                                 [&](size_t input_idx) -> GpuTensor* {
                                                     return &m_tensor_map.get_input_device(input_idx);
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

        m_last_profiling.clear();
        if (profiler) {
            const auto stats = m_allocator ? m_allocator->stats() : MetalMemoryStats{};
            profiler->set_memory_stats(stats);
            profiler->end_infer(cb);
            m_last_profiling = profiler->export_ov();
            if (cm) {
                cm->update_last_profiling_report(profiler->export_extended());
            }
        } else if (cm) {
            MetalProfilingReport empty;
            empty.level = ProfilingLevel::Off;
            cm->update_last_profiling_report(empty);
        }
    };

    run_pipeline();

    auto output_input_lookup = [&](size_t input_idx) -> GpuTensor* {
        if (!m_tensor_map.has_input_device(input_idx)) {
            return nullptr;
        }
        return &m_tensor_map.get_input_device(input_idx);
    };
    bind_outputs_common(
        get_outputs(),
        cm->get_runtime_model(),
        cm->node_to_stage(),
        cm->parameter_index(),
        pipeline,
        output_input_lookup,
        m_bound_remote_outputs,
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
            auto bound = bind_host_output(dev,
                                          info,
                                          host_override,
                                          GpuBackend::Metal,
                                          m_alloc_core,
                                          m_allocator.get(),
                                          cm->command_queue(),
                                          nullptr,
                                          nullptr,
                                          "GFX");
            m_tensor_map.bind_output_device(idx, bound.device_tensor);
            ov::ISyncInferRequest::set_tensor(get_outputs()[idx], ov::get_tensor_impl(bound.host_tensor));
        },
        /*allow_missing=*/false,
        /*allow_fallback_one=*/true,
        "GFX");

    if (const char* mem_flag = std::getenv("OV_GFX_MEM_STATS")) {
        (void)mem_flag;
        auto stats = m_allocator ? m_allocator->stats() : MetalMemoryStats{};
        double mb = 1024.0 * 1024.0;
        std::ostringstream oss;
        oss << "[GFX][mem] H2D=" << (stats.h2d_bytes / mb) << "MB "
            << "D2H=" << (stats.d2h_bytes / mb) << "MB "
            << "alloc=" << (stats.bytes_allocated_total / mb) << "MB "
            << "reuse_hits=" << stats.num_reuse_hits;
        std::cerr << oss.str() << std::endl;
    }

    if (m_allocator && cm) {
        cm->update_last_stats(m_allocator->stats());
    }
    }
}

std::vector<ov::ProfilingInfo> InferRequest::get_profiling_info() const {
    return m_last_profiling;
}

const std::shared_ptr<const CompiledModel> InferRequest::get_compiled_model_typed() const {
    return std::static_pointer_cast<const CompiledModel>(ov::ISyncInferRequest::get_compiled_model());
}

}  // namespace gfx_plugin
}  // namespace ov
