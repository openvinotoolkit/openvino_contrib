// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request.hpp"

#include <iostream>
#include <cstring>
#include <algorithm>

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

void InferRequest::infer() {
    auto cm = get_compiled_model_typed();
    OPENVINO_ASSERT(cm, "CompiledModel is null");

    std::vector<ov::Tensor> input_wrapped;
    input_wrapped.reserve(get_inputs().size());
    for (size_t idx = 0; idx < get_inputs().size(); ++idx) {
        ov::Tensor src;
        if (idx < m_bound_inputs.size() && m_bound_inputs[idx]) {
            src = m_bound_inputs[idx];
        } else {
            // Use the tensor owned by the base request (preallocated and filled by benchmark_app)
            auto impl = get_tensor(get_inputs()[idx]);
            if (!impl._ptr) {
                ov::Shape sh = get_inputs()[idx].get_partial_shape().is_static()
                    ? get_inputs()[idx].get_shape()
                    : ov::Shape{1};
                ov::Tensor fallback{get_inputs()[idx].get_element_type(), sh};
                ov::ISyncInferRequest::set_tensor(get_inputs()[idx], ov::get_tensor_impl(fallback));
                src = fallback;
            } else {
                src = ov::make_tensor(impl);
            }
            if (!src.data()) {
                ov::Shape sh = get_inputs()[idx].get_partial_shape().is_static()
                    ? get_inputs()[idx].get_shape()
                    : ov::Shape{1};
                src = ov::Tensor{get_inputs()[idx].get_element_type(), sh};
                ov::ISyncInferRequest::set_tensor(get_inputs()[idx], ov::get_tensor_impl(src));
            }
        }
        input_wrapped.emplace_back(std::move(src));
    }

    std::vector<ov::Tensor> output_wrapped;
    output_wrapped.reserve(get_outputs().size());
    for (size_t idx = 0; idx < get_outputs().size(); ++idx) {
        auto base_impl = get_tensor(get_outputs()[idx]);
        if (!base_impl._ptr) {
            ov::Shape osh = get_outputs()[idx].get_partial_shape().is_static()
                ? get_outputs()[idx].get_shape()
                : ov::Shape{1};
            ov::Tensor fallback{get_outputs()[idx].get_element_type(), osh};
            base_impl = ov::get_tensor_impl(fallback);
            ov::ISyncInferRequest::set_tensor(get_outputs()[idx], base_impl);
        }
        ov::Tensor base = ov::make_tensor(base_impl);
        if (!base.data()) {
            ov::Shape osh = get_outputs()[idx].get_partial_shape().is_static()
                ? get_outputs()[idx].get_shape()
                : ov::Shape{1};
            base = ov::Tensor{get_outputs()[idx].get_element_type(), osh};
            ov::ISyncInferRequest::set_tensor(get_outputs()[idx], ov::get_tensor_impl(base));
        }
        output_wrapped.emplace_back(base);
    }

    OPENVINO_ASSERT(cm->backend(), "Backend is null");
    if (auto mlir = dynamic_cast<MlirBackend*>(cm->backend())) {
        if (mlir->has_segment() && mlir->segment_io_is_model_io()) {
            std::vector<ov::Tensor> seg_inputs;
            seg_inputs.reserve(1);
            if (!input_wrapped.empty())
                seg_inputs.push_back(input_wrapped[0]);
            const auto& seg = mlir->get_segment();
            auto seg_outputs = mlir->run_segment(seg, seg_inputs);
            OPENVINO_ASSERT(!seg_outputs.empty(), "run_segment returned no outputs");
            for (size_t i = 0; i < std::min(seg_outputs.size(), output_wrapped.size()); ++i) {
                const auto& src = seg_outputs[i];
                auto& dst = output_wrapped[i];
                OPENVINO_ASSERT(src.get_byte_size() == dst.get_byte_size(),
                                "run_segment output size mismatch");
                std::memcpy(dst.data(), src.data(), src.get_byte_size());
            }
            return;
        }
    }
    cm->backend()->run(input_wrapped, output_wrapped);

    // Propagate potentially re-bound/reshaped outputs back to the base request storage.
    for (size_t idx = 0; idx < output_wrapped.size(); ++idx) {
        ov::ISyncInferRequest::set_tensor(get_outputs()[idx], ov::get_tensor_impl(output_wrapped[idx]));
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
