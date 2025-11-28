// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request.hpp"

#include "compiled_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
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
    for (const auto& output : get_outputs()) {
        allocate_tensor(output, [output](ov::SoPtr<ov::ITensor>& tensor) {
            tensor = ov::make_tensor(output.get_element_type(),
                                     output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape());
        });
    }
}

void InferRequest::infer() {
    check_tensors();
    auto cm = get_compiled_model_typed();
    OPENVINO_ASSERT(cm, "CompiledModel is null");

    std::vector<ov::Tensor> input_wrapped;
    input_wrapped.reserve(get_inputs().size());
    for (const auto& port : get_inputs()) {
        auto so = get_tensor(port);
        input_wrapped.emplace_back(ov::make_tensor(so));
    }

    std::vector<ov::Tensor> output_wrapped;
    output_wrapped.reserve(get_outputs().size());
    for (const auto& port : get_outputs()) {
        auto so = get_tensor(port);
        output_wrapped.emplace_back(ov::make_tensor(so));
    }

    mps_execute(cm->graph(), cm->input_tensors(), cm->output_tensors(), input_wrapped, output_wrapped);
}

const std::shared_ptr<const CompiledModel> InferRequest::get_compiled_model_typed() const {
    return std::static_pointer_cast<const CompiledModel>(ov::ISyncInferRequest::get_compiled_model());
}

}  // namespace metal_plugin
}  // namespace ov
