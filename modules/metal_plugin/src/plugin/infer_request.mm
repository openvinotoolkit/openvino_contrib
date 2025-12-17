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
#include "runtime/metal_logger.hpp"

namespace ov {
namespace metal_plugin {

InferRequest::InferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    // Allocate host tensors for all inputs/outputs using public factory
    for (const auto& input : get_inputs()) {
        allocate_tensor(input, [input](ov::SoPtr<ov::ITensor>& tensor) {
            tensor = ov::make_tensor(input.get_element_type(),
                                     input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape());
        });
    }
    m_bound_inputs.resize(get_inputs().size());
    for (const auto& output : get_outputs()) {
        allocate_tensor(output, [output](ov::SoPtr<ov::ITensor>& tensor) {
            tensor = ov::make_tensor(output.get_element_type(),
                                     output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape());
        });
    }
    if (auto cm = get_compiled_model_typed()) {
        m_buffer_manager = cm->buffer_manager();
    }
}

void InferRequest::set_input_tensor(const ov::Tensor& tensor) {
    // Single-input convenience: index 0
    set_input_tensor(0, tensor);
}

void InferRequest::set_input_tensor(size_t idx, const ov::Tensor& tensor) {
    if (idx >= m_bound_inputs.size())
        m_bound_inputs.resize(get_inputs().size());

    // Remote tensors are allowed for behavior tests but cannot expose host data; keep base bookkeeping only.
    auto impl = ov::get_tensor_impl(tensor);
    if (std::dynamic_pointer_cast<ov::IRemoteTensor>(impl._ptr)) {
        ov::ISyncInferRequest::set_tensor(get_inputs().at(idx), impl);
        m_bound_inputs[idx] = {};
        return;
    }

    // Cache a host view (no CPU copy)
    m_bound_inputs[idx] = tensor;
}

void InferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    // If remote tensor is provided, substitute with a host mirror to satisfy base checks.
    if (auto remote = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr)) {
        ov::Tensor host{remote->get_element_type(), remote->get_shape()};
        auto host_impl = ov::get_tensor_impl(host);
        ov::ISyncInferRequest::set_tensor(port, host_impl);
    } else {
        // Keep base bookkeeping (own storage) but ignore its data for Metal; we copy from the incoming view.
        ov::ISyncInferRequest::set_tensor(port, tensor);
    }

    auto& inputs = get_inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i] == port) {
            if (i >= m_bound_inputs.size())
                m_bound_inputs.resize(inputs.size());
            if (std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr)) {
                m_bound_inputs[i] = {};
            } else {
                ov::Tensor stored_view = ov::make_tensor(ov::ISyncInferRequest::get_tensor(port));
                m_bound_inputs[i] = stored_view;
            }
            break;
        }
    }
}

ov::SoPtr<ov::ITensor> InferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    auto found = find_port(port);
    if (found.found() && found.is_output()) {
        size_t idx = found.idx;
        if (!m_buffer_manager) {
            if (auto cm = get_compiled_model_typed())
                m_buffer_manager = cm->buffer_manager();
        }
        if (m_buffer_manager && m_tensor_map.has_output_device(idx)) {
            const auto& dev = m_tensor_map.get_output_device(idx);
            if (dev.buf.storage_mode == static_cast<uint32_t>(MTLStorageModeShared)) {
                id<MTLBuffer> buf = static_cast<id<MTLBuffer>>(dev.buf.buffer);
                void* ptr = buf ? [buf contents] : nullptr;
                if (!ptr) {
                    OPENVINO_THROW("METAL: shared output buffer has no CPU pointer");
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
            OPENVINO_THROW("METAL: output buffer is private (no CPU access, no copies)");
        }
    }
    return ov::ISyncInferRequest::get_tensor(port);
}

ov::Tensor InferRequest::get_output_tensor(size_t idx) const {
    auto so = get_tensor(get_outputs().at(idx));
    return ov::make_tensor(so);
}

void InferRequest::infer() {
    auto cm = get_compiled_model_typed();
    OPENVINO_ASSERT(cm, "CompiledModel is null");

    if (!m_buffer_manager) {
        m_buffer_manager = cm->buffer_manager();
    }
    if (!m_buffer_manager) {
        m_buffer_manager = cm->backend()->create_buffer_manager();
    }
    OPENVINO_ASSERT(m_buffer_manager, "MetalBufferManager is null");

    m_tensor_map.reset_inference();
    if (m_buffer_manager) {
        m_buffer_manager->reset_stats();
        m_buffer_manager->reset_inference_pool();
    }

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
        ov::Tensor src = ensure_input_tensor(idx);
        if (!src) {
            ov::Shape sh = get_inputs()[idx].get_partial_shape().is_static()
                               ? get_inputs()[idx].get_shape()
                               : ov::Shape{1};
            src = ov::Tensor{get_inputs()[idx].get_element_type(), sh};
        }
        host_inputs.emplace_back(src);
        // Bind host -> device (Shared buffer to allow memcpy).
        m_tensor_map.bind_input(idx, src, *m_buffer_manager, /*shared=*/true);
    }

    auto prepare_output_host = [&](size_t idx) -> ov::Tensor {
        auto base_impl = get_tensor(get_outputs()[idx]);
        if (!base_impl._ptr) {
            ov::Shape osh = get_outputs()[idx].get_partial_shape().is_static()
                                ? get_outputs()[idx].get_shape()
                                : ov::Shape{1};
            ov::Tensor fallback{get_outputs()[idx].get_element_type(), osh};
            base_impl = ov::get_tensor_impl(fallback);
            ov::ISyncInferRequest::set_tensor(get_outputs()[idx], base_impl);
            return fallback;
        }
        ov::Tensor base = ov::make_tensor(base_impl);
        if (!base.data()) {
            ov::Shape osh = get_outputs()[idx].get_partial_shape().is_static()
                                ? get_outputs()[idx].get_shape()
                                : ov::Shape{1};
            base = ov::Tensor{get_outputs()[idx].get_element_type(), osh};
            ov::ISyncInferRequest::set_tensor(get_outputs()[idx], ov::get_tensor_impl(base));
        }
        return base;
    };

    auto run_pipeline = [&]() -> bool {
        if (!cm->op_pipeline_enabled() || !cm->op_pipeline_built() || cm->op_pipeline_size() == 0)
            return false;
        auto& pipeline = cm->pipeline_mutable();
        const auto& node_map = cm->node_to_stage();
        const auto& param_map = cm->parameter_index();

        // Allocate outputs and bind inputs for each stage.
        for (auto& stage : pipeline) {
            for (size_t oi = 0; oi < stage.outputs.size(); ++oi) {
                auto& out_ref = stage.outputs[oi];
                if (!out_ref->buf.valid()) {
                    const auto& et = stage.node->get_output_element_type(oi);
                    size_t bytes = et.size();
                    if (out_ref->shape.empty() && stage.node->get_output_partial_shape(oi).is_static()) {
                        out_ref->shape = stage.node->get_output_shape(oi);
                    }
                    for (auto d : out_ref->shape) bytes *= d;
                    out_ref->buf = m_buffer_manager->allocate(bytes, et, /*persistent=*/false, /*storageModePrivate=*/true);
                }
            }
        }

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
            stage.op->set_inputs(resolved);
            stage.op->init(m_buffer_manager.get());
            stage.op->execute();
        }

        // Bind model outputs to device tensors from pipeline
        for (size_t out_idx = 0; out_idx < get_outputs().size(); ++out_idx) {
            auto res_node = get_outputs()[out_idx].get_node();
            auto src_node = res_node->input_value(0).get_node_shared_ptr();
            auto it = node_map.find(src_node.get());
            if (it != node_map.end()) {
                size_t src_port = res_node->input_value(0).get_index();
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
            // Fallback to legacy device path if pipeline is incomplete.
            return false;
        }
        return true;
    };

    bool device_ok = run_pipeline();
    if (!device_ok) {
        OPENVINO_ASSERT(cm->backend(), "Backend is null");
        if (auto mlir = dynamic_cast<MlirBackend*>(cm->backend())) {
            device_ok = mlir->run_device(m_tensor_map, *m_buffer_manager);
        }
    }

    OPENVINO_ASSERT(device_ok, "METAL: device execution failed and CPU fallback is disabled");

    // Device path: ensure base outputs carry correct shapes but keep data on device.
    for (size_t idx = 0; idx < get_outputs().size(); ++idx) {
        if (!m_tensor_map.has_output_device(idx))
            continue;
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
        if (dev.buf.storage_mode == static_cast<uint32_t>(MTLStorageModeShared)) {
            id<MTLBuffer> buf = static_cast<id<MTLBuffer>>(dev.buf.buffer);
            void* ptr = buf ? [buf contents] : nullptr;
            OPENVINO_ASSERT(ptr, "METAL: shared output buffer has no CPU pointer");
            ov::Tensor view{logical, shape, ptr};
            ov::ISyncInferRequest::set_tensor(get_outputs()[idx], ov::get_tensor_impl(view));
        } else {
            ov::Tensor shadow{logical, shape};
            ov::ISyncInferRequest::set_tensor(get_outputs()[idx], ov::get_tensor_impl(shadow));
        }
    }

    if (const char* mem_flag = std::getenv("OV_METAL_MEM_STATS")) {
        (void)mem_flag;
        auto stats = m_buffer_manager->stats();
        double mb = 1024.0 * 1024.0;
        std::ostringstream oss;
        oss << "[METAL][mem] H2D=" << (stats.h2d_bytes / mb) << "MB "
            << "D2H=" << (stats.d2h_bytes / mb) << "MB "
            << "alloc=" << (stats.alloc_bytes / mb) << "MB "
            << "reused=" << (stats.reused_bytes / mb) << "MB";
        std::cerr << oss.str() << std::endl;
    }
}

std::vector<ov::ProfilingInfo> InferRequest::get_profiling_info() const {
    auto cm = get_compiled_model_typed();
    if (!cm || !cm->backend())
        return {};
    return cm->backend()->get_profiling_info();
}

const std::shared_ptr<const CompiledModel> InferRequest::get_compiled_model_typed() const {
    return std::static_pointer_cast<const CompiledModel>(ov::ISyncInferRequest::get_compiled_model());
}

}  // namespace metal_plugin
}  // namespace ov
