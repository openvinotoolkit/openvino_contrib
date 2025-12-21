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
#include "runtime/metal_logger.hpp"
#include "runtime/metal_memory.hpp"
#include "runtime/profiling/metal_profiler.hpp"
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
        m_alloc_core = &cm->allocator_core();
        m_caps = cm->device_caps();
        m_heaps = std::make_unique<MetalHeapPool>(*m_alloc_core);
        m_freelist = std::make_unique<MetalFreeList>();
        m_staging = std::make_unique<MetalStagingPool>(*m_alloc_core);
        m_allocator = std::make_unique<MetalAllocator>(*m_alloc_core, *m_heaps, *m_freelist, *m_staging, m_caps);
    }
}

InferRequest::~InferRequest() = default;

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
        auto metal_remote = std::dynamic_pointer_cast<GfxRemoteTensor>(remote);
        OPENVINO_ASSERT(metal_remote, "GFX: remote tensor type mismatch");
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

        auto ensure_input_tensor = [&](size_t idx) -> ov::Tensor {
            if (idx < m_bound_inputs.size() && m_bound_inputs[idx]) {
                return m_bound_inputs[idx];
            }
            auto impl = get_tensor(get_inputs()[idx]);
            ov::Tensor src;
            if (!impl._ptr) {
                ov::Shape sh = get_inputs()[idx].get_partial_shape().is_static()
                                   ? get_inputs()[idx].get_shape()
                                   : ov::Shape{1};
                src = ov::Tensor{get_inputs()[idx].get_element_type(), sh};
                ov::ISyncInferRequest::set_tensor(get_inputs()[idx], ov::get_tensor_impl(src));
            } else {
                src = ov::make_tensor(impl);
            }
            if (!src || !src.data()) {
                ov::Shape sh = get_inputs()[idx].get_partial_shape().is_static()
                                   ? get_inputs()[idx].get_shape()
                                   : ov::Shape{1};
                src = ov::Tensor{get_inputs()[idx].get_element_type(), sh};
                ov::ISyncInferRequest::set_tensor(get_inputs()[idx], ov::get_tensor_impl(src));
            }
            return src;
        };

        std::vector<ov::Tensor> host_inputs;
        host_inputs.reserve(get_inputs().size());
        for (size_t idx = 0; idx < get_inputs().size(); ++idx) {
            if (idx < m_bound_remote_inputs.size() && m_bound_remote_inputs[idx]) {
            m_tensor_map.bind_input_device(idx, m_bound_remote_inputs[idx]->metal_tensor());
            continue;
        }
        ov::Tensor src = ensure_input_tensor(idx);
        if (!src) {
            ov::Shape sh = get_inputs()[idx].get_partial_shape().is_static()
                               ? get_inputs()[idx].get_shape()
                               : ov::Shape{1};
            src = ov::Tensor{get_inputs()[idx].get_element_type(), sh};
        }
        host_inputs.emplace_back(src);
        // Bind host -> device using zero-copy shared buffer (no memcpy).
        m_tensor_map.bind_input(idx, src, *m_alloc_core);
    }

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

    auto run_pipeline = [&]() {
        struct DebugFilter {
            bool enabled = false;
            bool all = false;
            std::vector<std::string> tokens;
        } dbg;
        if (const char* env = std::getenv("OV_GFX_DEBUG_TENSORS")) {
            std::string spec = env;
            if (!spec.empty()) {
                dbg.enabled = true;
                if (spec == "all") {
                    dbg.all = true;
                } else {
                    size_t pos = 0;
                    while (pos < spec.size()) {
                        size_t comma = spec.find(',', pos);
                        if (comma == std::string::npos)
                            comma = spec.size();
                        std::string tok = spec.substr(pos, comma - pos);
                        if (!tok.empty())
                            dbg.tokens.push_back(tok);
                        pos = comma + 1;
                    }
                }
            }
        }
        auto match_debug = [&](const std::string& name, const std::string& type) -> bool {
            if (!dbg.enabled)
                return false;
            if (dbg.all)
                return true;
            for (const auto& tok : dbg.tokens) {
                if (name.find(tok) != std::string::npos || type.find(tok) != std::string::npos)
                    return true;
            }
            return false;
        };
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

        struct InferStage {
            std::shared_ptr<const ov::Node> node;
            std::unique_ptr<MetalOp> op;
            std::vector<std::unique_ptr<MetalTensor>> outputs;
            std::vector<bool> output_is_model_output;
            std::vector<PipelineStageDesc::InputLink> inputs;
        };

        std::vector<InferStage> pipeline;
        pipeline.reserve(descs.size());
        const bool profiling_enabled = (profiler != nullptr);
        for (size_t stage_id = 0; stage_id < descs.size(); ++stage_id) {
            const auto& desc = descs[stage_id];
            InferStage stage;
            stage.node = desc.node;
            stage.op = MetalOpFactory::clone(*desc.op);
            OPENVINO_ASSERT(stage.op, "GFX: failed to clone op for stage ", desc.node->get_friendly_name());
            stage.inputs = desc.inputs;
            stage.outputs.reserve(desc.outputs.size());
            stage.output_is_model_output.reserve(desc.outputs.size());
            for (const auto& out_desc : desc.outputs) {
                auto out_tensor = std::make_unique<MetalTensor>();
                out_tensor->shape = out_desc.shape;
                out_tensor->expected_type = out_desc.type;
                stage.outputs.emplace_back(std::move(out_tensor));
                stage.output_is_model_output.push_back(out_desc.is_model_output);
            }
            if (stage.outputs.size() == 1) {
                stage.op->set_output(stage.outputs[0].get());
            } else {
                stage.op->set_outputs(stage.outputs);
            }
            stage.op->init(cm->const_manager().get());
            stage.op->enable_profiling(profiling_enabled);
            if (profiling_enabled) {
                const std::string node_name =
                    stage.node ? stage.node->get_friendly_name() : stage.op->name();
                const std::string node_type =
                    stage.node ? stage.node->get_type_name() : stage.op->type();
                stage.op->set_profiler(profiler,
                                       static_cast<uint32_t>(stage_id),
                                       node_name,
                                       node_type);
            }
            pipeline.emplace_back(std::move(stage));
        }

        // Bind remote outputs to pipeline buffers before allocations.
        if (!m_bound_remote_outputs.empty()) {
            for (size_t out_idx = 0; out_idx < get_outputs().size(); ++out_idx) {
                if (out_idx >= m_bound_remote_outputs.size() || !m_bound_remote_outputs[out_idx]) {
                    continue;
                }
                auto res_node = get_outputs()[out_idx].get_node();
                auto src_node = res_node->input_value(0).get_node_shared_ptr();
                if (auto it = node_map.find(src_node.get()); it != node_map.end()) {
                    size_t src_port = res_node->input_value(0).get_index();
                    auto& outs = pipeline[it->second].outputs;
                    OPENVINO_ASSERT(src_port < outs.size(), "GFX: remote output port out of range");
                    auto& dst = outs[src_port];
                    dst->buf = m_bound_remote_outputs[out_idx]->metal_tensor().buf;
                    dst->shape = m_bound_remote_outputs[out_idx]->metal_tensor().shape;
                    dst->expected_type = m_bound_remote_outputs[out_idx]->metal_tensor().expected_type;
                    continue;
                }
                if (auto pit = param_map.find(src_node.get()); pit != param_map.end()) {
                    const size_t input_idx = pit->second;
                    if (input_idx < m_bound_remote_inputs.size() && m_bound_remote_inputs[input_idx]) {
                        auto in_buf = m_bound_remote_inputs[input_idx]->metal_tensor().buf.buffer;
                        auto out_buf = m_bound_remote_outputs[out_idx]->metal_tensor().buf.buffer;
                        OPENVINO_ASSERT(in_buf == out_buf,
                                        "GFX: remote output must alias remote input for passthrough outputs");
                        continue;
                    }
                    OPENVINO_THROW("GFX: remote output cannot be bound to non-remote input passthrough");
                }
                OPENVINO_THROW("GFX: failed to bind remote output ", out_idx, " (pipeline incomplete)");
            }
        }

        // Allocate outputs (reuse buffers across iterations via handles).
        if (m_stage_output_handles.size() != pipeline.size()) {
            m_stage_output_handles.assign(pipeline.size(), {});
        }
        for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
            auto& stage = pipeline[stage_idx];
            auto& handles = m_stage_output_handles[stage_idx];
            if (handles.size() < stage.outputs.size()) {
                handles.resize(stage.outputs.size());
            }
            for (size_t oi = 0; oi < stage.outputs.size(); ++oi) {
                auto& out_ref = stage.outputs[oi];
                const bool is_model_output = (oi < stage.output_is_model_output.size()) &&
                                             stage.output_is_model_output[oi];
                out_ref->prefer_private = !is_model_output;
                if (!out_ref->buf.valid()) {
                    if (out_ref->shape.empty() && stage.node->get_output_partial_shape(oi).is_static()) {
                        out_ref->shape = stage.node->get_output_shape(oi);
                    }
                    if (out_ref->shape.empty()) {
                        continue;  // let op allocate at runtime
                    }
                    const auto& et = stage.node->get_output_element_type(oi);
                    size_t bytes = et.size();
                    for (auto d : out_ref->shape) bytes *= d;
                    BufferDesc desc;
                    desc.bytes = bytes;
                    desc.type = et;
                    desc.usage = BufferUsage::Intermediate;
                    desc.storage = out_ref->prefer_private ? MetalStorage::Private : MetalStorage::Shared;
                    out_ref->buf = m_allocator->ensure_handle(handles[oi], desc, /*persistent=*/false);
                }
            }
        }

        if (profiler) {
            const size_t expected_samples = pipeline.size() * 4;
            profiler->begin_infer(expected_samples);
        }

        // Create a single command buffer for the entire pipeline.
        id<MTLCommandQueue> cq = static_cast<id<MTLCommandQueue>>(cm->command_queue());
        OPENVINO_ASSERT(cq, "GFX: command queue is null");
        id<MTLCommandBuffer> cb = [cq commandBuffer];

        for (const auto& stage : pipeline) {
            std::vector<MetalTensor*> resolved;
            resolved.reserve(stage.inputs.size());
            for (const auto& link : stage.inputs) {
                if (!link.node) {
                    resolved.push_back(nullptr);
                    continue;
                }
                if (auto itp = param_map.find(link.node.get()); itp != param_map.end()) {
                    resolved.push_back(&m_tensor_map.get_input_device(itp->second));
                    continue;
                }
                if (auto it = node_map.find(link.node.get()); it != node_map.end()) {
                    auto& src_stage = pipeline[it->second];
                    MetalTensor* tensor = nullptr;
                    if (link.port < src_stage.outputs.size()) {
                        tensor = src_stage.outputs[link.port].get();
                    }
                    resolved.push_back(tensor);
                    continue;
                }
                resolved.push_back(nullptr);  // constants handled inside ops
            }
            if (metal_log_debug_enabled() || metal_safe_debug_enabled()) {
                const std::string node_name =
                    stage.node ? stage.node->get_friendly_name() : stage.op->name();
                const std::string node_type =
                    stage.node ? stage.node->get_type_name() : stage.op->type();
                if (metal_log_debug_enabled()) {
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
            stage.op->set_inputs(resolved);
            stage.op->execute(cb);

            if (dbg.enabled) {
                const std::string node_name =
                    stage.node ? stage.node->get_friendly_name() : stage.op->name();
                const std::string node_type =
                    stage.node ? stage.node->get_type_name() : stage.op->type();
                if (match_debug(node_name, node_type)) {
                    for (size_t oi = 0; oi < stage.outputs.size(); ++oi) {
                        MetalTensor* src = stage.outputs[oi].get();
                        if (!src || !src->buf.valid())
                            continue;
                        ov::Shape shape = src->shape;
                        if (shape.empty() && stage.node &&
                            stage.node->get_output_partial_shape(oi).is_static()) {
                            shape = stage.node->get_output_shape(oi);
                        }
                        if (shape.empty())
                            continue;
                        ov::element::Type logical =
                            src->expected_type == ov::element::dynamic ? src->buf.type : src->expected_type;
                        size_t bytes = logical.size();
                        for (auto d : shape) bytes *= d;
                        BufferDesc desc;
                        desc.bytes = bytes;
                        desc.type = logical;
                        desc.usage = BufferUsage::Temp;
                        desc.storage = MetalStorage::Shared;
                        desc.cpu_read = true;
                        desc.cpu_write = true;
                        MetalBuffer snap = m_allocator->allocate(desc, /*persistent=*/false);
                        id<MTLCommandQueue> cq = static_cast<id<MTLCommandQueue>>(cm->command_queue());
                        OPENVINO_ASSERT(cq, "GFX: command queue is null");
                        id<MTLCommandBuffer> blit_cb = [cq commandBuffer];
                        id<MTLBlitCommandEncoder> blit = [blit_cb blitCommandEncoder];
                        [blit copyFromBuffer:static_cast<id<MTLBuffer>>(src->buf.buffer)
                               sourceOffset:0
                                   toBuffer:static_cast<id<MTLBuffer>>(snap.buffer)
                          destinationOffset:0
                                        size:bytes];
                        [blit endEncoding];
                        [blit_cb commit];
                        [blit_cb waitUntilCompleted];
                        id<MTLBuffer> buf = static_cast<id<MTLBuffer>>(snap.buffer);
                        void* ptr = buf ? [buf contents] : nullptr;
                        if (ptr) {
                            ov::Tensor view{logical, shape, ptr};
                            std::string tag = node_name + ":" + std::to_string(oi);
                            m_debug_tensors.emplace_back(tag, view);
                            m_debug_buffers.emplace_back(std::move(snap));
                        } else {
                            if (snap.valid()) {
                                m_allocator->release(std::move(snap));
                            }
                        }
                    }
                }
            }
        }

        [cb commit];
        [cb waitUntilCompleted];

        // Bind model outputs to device tensors from pipeline
        const auto runtime_model = cm->get_runtime_model();
        const auto& public_outputs = get_outputs();
        const auto runtime_results = runtime_model ? runtime_model->get_results() : ov::ResultVector{};
        const bool use_runtime_results = runtime_results.size() == public_outputs.size();
        for (size_t out_idx = 0; out_idx < public_outputs.size(); ++out_idx) {
            std::shared_ptr<const ov::Node> src_node;
            size_t src_port = 0;
            if (use_runtime_results) {
                auto res_node = runtime_results[out_idx];
                src_node = res_node->input_value(0).get_node_shared_ptr();
                src_port = res_node->input_value(0).get_index();
            } else {
                auto res_node = public_outputs[out_idx].get_node();
                src_node = res_node->input_value(0).get_node_shared_ptr();
                src_port = res_node->input_value(0).get_index();
            }
            auto it = node_map.find(src_node.get());
            if (it != node_map.end()) {
                auto& outs = pipeline[it->second].outputs;
                if (src_port < outs.size() && outs[src_port]) {
                    m_tensor_map.bind_output_device(out_idx, *outs[src_port]);
                    continue;
                }
            }
            if (auto pit = param_map.find(src_node.get()); pit != param_map.end()) {
                // Direct passthrough from model input.
                m_tensor_map.bind_output_device(out_idx, m_tensor_map.get_input_device(pit->second));
                continue;
            }
            OPENVINO_THROW("GFX: failed to bind output ", out_idx, " (pipeline incomplete)");
        }

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

    // Outputs: device-only by default; optional host-visible shared buffers when enabled.
    auto ensure_host_visible = [&](const MetalTensor& src,
                                   ov::element::Type logical_type,
                                   const ov::Shape& shape,
                                   const ov::Tensor* host_override) {
        size_t bytes = logical_type.size();
        for (auto d : shape) bytes *= d;

        if (host_override && *host_override) {
            OPENVINO_ASSERT(host_override->get_element_type() == logical_type,
                            "GFX: output tensor type mismatch");
            OPENVINO_ASSERT(host_override->get_shape() == shape,
                            "GFX: output tensor shape mismatch");
            OPENVINO_ASSERT(host_override->data(), "GFX: output tensor has null data");
            MetalBuffer shared = m_alloc_core->wrap_shared(host_override->data(), bytes, logical_type);
            if (src.buf.buffer != shared.buffer) {
                id<MTLCommandQueue> cq = static_cast<id<MTLCommandQueue>>(cm->command_queue());
                OPENVINO_ASSERT(cq, "GFX: command queue is null");
                id<MTLCommandBuffer> blit_cb = [cq commandBuffer];
                id<MTLBlitCommandEncoder> blit = [blit_cb blitCommandEncoder];
                [blit copyFromBuffer:static_cast<id<MTLBuffer>>(src.buf.buffer)
                       sourceOffset:0
                           toBuffer:static_cast<id<MTLBuffer>>(shared.buffer)
                  destinationOffset:0
                                size:bytes];
                [blit endEncoding];
                [blit_cb commit];
                [blit_cb waitUntilCompleted];
            }
            MetalTensor out = src;
            out.buf = shared;
            out.expected_type = logical_type;
            out.shape = shape;
            out.prefer_private = false;
            return out;
        }

        if (src.buf.storage_mode == static_cast<uint32_t>(MTLStorageModeShared)) {
            return src;
        }
        // Allocate shared buffer and copy on GPU (no CPU copies).
        BufferDesc desc;
        desc.bytes = bytes;
        desc.type = logical_type;
        desc.usage = BufferUsage::IO;
        desc.storage = MetalStorage::Shared;
        desc.cpu_read = true;
        desc.cpu_write = true;
        MetalBuffer shared = m_allocator->allocate(desc, /*persistent=*/false);
        id<MTLCommandQueue> cq = static_cast<id<MTLCommandQueue>>(cm->command_queue());
        OPENVINO_ASSERT(cq, "GFX: command queue is null");
        id<MTLCommandBuffer> blit_cb = [cq commandBuffer];
        id<MTLBlitCommandEncoder> blit = [blit_cb blitCommandEncoder];
        [blit copyFromBuffer:static_cast<id<MTLBuffer>>(src.buf.buffer)
               sourceOffset:0
                   toBuffer:static_cast<id<MTLBuffer>>(shared.buffer)
          destinationOffset:0
                        size:bytes];
        [blit endEncoding];
        [blit_cb commit];
        [blit_cb waitUntilCompleted];
        MetalTensor out = src;
        out.buf = shared;
        out.expected_type = logical_type;
        out.shape = shape;
        out.prefer_private = false;
        return out;
    };
    for (size_t idx = 0; idx < get_outputs().size(); ++idx) {
        if (idx < m_bound_remote_outputs.size() && m_bound_remote_outputs[idx]) {
            ov::ISyncInferRequest::set_tensor(get_outputs()[idx],
                                              ov::SoPtr<ov::ITensor>{m_bound_remote_outputs[idx], nullptr});
            continue;
        }
        if (!m_tensor_map.has_output_device(idx)) {
            OPENVINO_THROW("GFX: output device tensor missing (pipeline incomplete)");
        }
        const auto& dev = m_tensor_map.get_output_device(idx);
        ov::element::Type logical = dev.expected_type == ov::element::dynamic ? dev.buf.type : dev.expected_type;
        ov::Shape shape = dev.shape;
        if (shape.empty()) {
            const auto& out = get_outputs()[idx];
            if (out.get_partial_shape().is_static())
                shape = out.get_shape();
            else
                shape = ov::Shape{1};
        }
        const ov::Tensor* host_override = nullptr;
        if (idx < m_bound_output_hosts.size() && m_bound_output_hosts[idx]) {
            host_override = &m_bound_output_hosts[idx];
        }
        MetalTensor host_dev = ensure_host_visible(dev, logical, shape, host_override);
        m_tensor_map.bind_output_device(idx, host_dev);
        if (host_override && *host_override) {
            ov::ISyncInferRequest::set_tensor(get_outputs()[idx], ov::get_tensor_impl(*host_override));
        } else {
            id<MTLBuffer> buf = static_cast<id<MTLBuffer>>(host_dev.buf.buffer);
            void* ptr = buf ? [buf contents] : nullptr;
            OPENVINO_ASSERT(ptr, "GFX: shared output buffer has no CPU pointer");
            ov::Tensor view{logical, shape, ptr};
            ov::ISyncInferRequest::set_tensor(get_outputs()[idx], ov::get_tensor_impl(view));
        }
    }

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
