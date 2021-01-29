// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/reverse_sequence.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReverseSequence& node) {
    auto make = [&] (auto refFunction) {
        return MakeConversion(refFunction,
                              node.input(0),
                              node.output(0),
                              node.get_input_shape(0),
                              node.get_batch_axis(),
                              node.get_sequence_axis(),
                              node.input(1));
    };

    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::u8 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::reverse_sequence<std::uint8_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::reverse_sequence<std::uint8_t, std::int64_t>);
        case ngraph::element::Type_t::i16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::reverse_sequence<std::int16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::reverse_sequence<std::int16_t, std::int64_t>);
        case ngraph::element::Type_t::u16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::reverse_sequence<std::uint16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::reverse_sequence<std::uint16_t, std::int64_t>);
        case ngraph::element::Type_t::u32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::reverse_sequence<std::uint32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::reverse_sequence<std::uint32_t, std::int64_t>);
        case ngraph::element::Type_t::i32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::reverse_sequence<std::int32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::reverse_sequence<std::int32_t, std::int64_t>);
        case ngraph::element::Type_t::i64 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::reverse_sequence<std::int64_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::reverse_sequence<std::int64_t, std::int64_t>);
        case ngraph::element::Type_t::f16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::reverse_sequence<half_float::half, std::int32_t>);
            }
            return make(ngraph::runtime::reference::reverse_sequence<half_float::half, std::int64_t>);
        case ngraph::element::Type_t::f32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::reverse_sequence<float, std::int32_t>);
            }
            return make(ngraph::runtime::reference::reverse_sequence<float, std::int64_t>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

}  //  namespace ArmPlugin
