// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include <arm_compute/runtime/NEON/functions/NECopy.h>
#include <arm_compute/runtime/NEON/functions/NEConcatenateLayer.h>
#include <ngraph/runtime/reference/concat.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template <typename T>
static void wrap_concat(const std::vector<Argument<Tensor*>>& inputs,
                        T* out,
                        const std::vector<ngraph::Shape>& inp_shapes,
                        const ngraph::Shape& out_shape,
                        int64_t axis,
                        size_t elem_size) {
    std::vector<const char*> char_inputs;
    for (const auto& inp : inputs) {
        char_inputs.push_back(reinterpret_cast<const char*>(static_cast<const T*>(inp)));
    }
    ngraph::runtime::reference::concat(char_inputs,
                                       reinterpret_cast<char*>(out),
                                       inp_shapes,
                                       out_shape,
                                       axis,
                                       elem_size);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Concat& node) {
    auto make = [&] (auto refFunction) {
        std::vector<ngraph::Shape> in_shapes;
        for (const auto& input : node.inputs()) {
            in_shapes.push_back(input.get_shape());
        }
        return this->MakeConversion(refFunction,
                                    node.inputs(),
                                    node.output(0),
                                    in_shapes,
                                    node.get_output_shape(0),
                                    node.get_axis(),
                                    node.get_input_element_type(0).size());
    };

    return CallSwitch(
        AP_WRAP(make, wrap_concat),
        node.input(0), allTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmConcat& node) {
    if (node.get_input_size() == 1) {
        return MakeConversion<arm_compute::NECopy>(node.input(0), node.output(0));
    }

    return MakeConversion<arm_compute::NEConcatenateLayer>(node.inputs(),
                                                           node.output(0),
                                                           AxisCast(node.get_axis(), node.get_input_shape(0).size()));
}
} // namespace ArmPlugin