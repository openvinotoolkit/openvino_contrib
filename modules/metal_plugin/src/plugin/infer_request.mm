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

static ov::Tensor deep_copy_tensor(const ov::Tensor& src) {
    ov::Tensor dst{src.get_element_type(), src.get_shape()};
    std::memcpy(dst.data(), src.data(), src.get_byte_size());
    return dst;
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

    // Cache a host copy before delegating to base API
    m_bound_inputs[idx] = deep_copy_tensor(tensor);
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
                m_bound_inputs[i] = deep_copy_tensor(stored_view);
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
            auto& host = m_tensor_map.get_or_create_host_for_output(idx, *m_buffer_manager);
            return ov::get_tensor_impl(host);
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
    m_buffer_manager->reset_stats();
    m_buffer_manager->reset_inference_pool();

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

    bool device_ok = false;
    OPENVINO_ASSERT(cm->backend(), "Backend is null");
    if (auto mlir = dynamic_cast<MlirBackend*>(cm->backend())) {
        device_ok = mlir->run_device(m_tensor_map, *m_buffer_manager);
    }

    OPENVINO_ASSERT(device_ok, "METAL: device execution failed and CPU fallback is disabled");

    // Device path: ensure base outputs carry correct shapes but keep data on device.
    for (size_t idx = 0; idx < get_outputs().size(); ++idx) {
        if (!m_tensor_map.has_output_device(idx))
            continue;
        const auto& dev = m_tensor_map.get_output_device(idx);
        ov::element::Type logical = dev.expected_type == ov::element::dynamic ? dev.buf.type : dev.expected_type;
        ov::Tensor shadow{logical, dev.shape};
        ov::ISyncInferRequest::set_tensor(get_outputs()[idx], ov::get_tensor_impl(shadow));
        if (std::getenv("METAL_DEBUG_DUMP_OUTPUT")) {
            ov::Tensor tmp = m_buffer_manager->copy_to_host(dev);
            if (tmp && tmp.get_size() > 0) {
                fprintf(stderr, "[METAL][dbg] out%zu first=%f (shape=%zu...)\n",
                        idx, tmp.data<const float>()[0], tmp.get_size());
            }
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
