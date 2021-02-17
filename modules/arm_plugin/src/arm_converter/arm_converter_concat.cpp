// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include <arm_compute/runtime/NEON/functions/NECopy.h>
#include <arm_compute/runtime/NEON/functions/NEConcatenateLayer.h>
#include <ngraph/runtime/reference/concat.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template <typename T>
static void wrap_concat(const std::vector<Argument<arm_compute::ITensor*>>& inputs,
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

template<> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::Concat& node) {
    if (node.get_shape().size() > 4) {
        THROW_IE_EXCEPTION << "Unsupported Concat with num dimensions > 4";
    }
    if (node.get_input_size() == 1) {
        return MakeConversion<arm_compute::NECopy>(node.input(0), node.output(0));
    }

    auto make = [&] (auto refFunction) {
        std::vector<ngraph::Shape> in_shapes;
        for (const auto& input : node.inputs()) {
            in_shapes.push_back(input.get_shape());
        }
        return MakeConversion(refFunction,
                              node.inputs(),
                              node.output(0),
                              in_shapes,
                              node.get_output_shape(0),
                              node.get_axis(),
                              node.input(0).get_element_type().size());
    };

    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::u8  : return make(wrap_concat<std::uint8_t>);
        case ngraph::element::Type_t::i16 : return make(wrap_concat<std::int16_t>);
        case ngraph::element::Type_t::u16 : return make(wrap_concat<std::uint16_t>);
        case ngraph::element::Type_t::i32 : return make(wrap_concat<std::int32_t>);
        case ngraph::element::Type_t::i64 : return make(wrap_concat<std::int64_t>);
        default: return MakeConversion<arm_compute::NEConcatenateLayer>(node.inputs(),
                                                                        node.output(0),
                                                                        AxisCast(node.get_axis(), node.get_input_shape(0).size()));
    }
}
} // namespace ArmPlugin