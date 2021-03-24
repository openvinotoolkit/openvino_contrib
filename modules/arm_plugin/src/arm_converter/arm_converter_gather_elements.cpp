// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/gather_elements.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::GatherElements& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_output_shape(0),
                                    node.get_axis());
    };

    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::u8 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_elements<std::uint8_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_elements<std::uint8_t, std::int64_t>);
        case ngraph::element::Type_t::i16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_elements<std::int16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_elements<std::int16_t, std::int64_t>);
        case ngraph::element::Type_t::u16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_elements<std::uint16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_elements<std::uint16_t, std::int64_t>);
        case ngraph::element::Type_t::u32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_elements<std::uint32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_elements<std::uint32_t, std::int64_t>);
        case ngraph::element::Type_t::i32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_elements<std::int32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_elements<std::int32_t, std::int64_t>);
        case ngraph::element::Type_t::i64 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_elements<std::int64_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_elements<std::int64_t, std::int64_t>);
        case ngraph::element::Type_t::f16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_elements<ngraph::float16, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_elements<ngraph::float16, std::int64_t>);
        case ngraph::element::Type_t::f32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_elements<float, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_elements<float, std::int64_t>);
        default: IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}

}  //  namespace ArmPlugin
