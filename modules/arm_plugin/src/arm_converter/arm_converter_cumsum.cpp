// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/cum_sum.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::CumSum& node) {
    auto make = [&] (auto refFunction) {
        return MakeConversion(refFunction, node.input(0), node.input(1), node.output(0),
                              node.get_input_shape(0), node.is_exclusive(), node.is_reverse());
    };
    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::u8 :
            switch (node.get_input_element_type(1)) {
                case ngraph::element::Type_t::u8  : return make(ngraph::runtime::reference::cumsum<std::uint8_t, std::uint8_t>);
                case ngraph::element::Type_t::i16 : return make(ngraph::runtime::reference::cumsum<std::uint8_t, std::int16_t>);
                case ngraph::element::Type_t::u16 : return make(ngraph::runtime::reference::cumsum<std::uint8_t, std::uint16_t>);
                case ngraph::element::Type_t::i32 : return make(ngraph::runtime::reference::cumsum<std::uint8_t, std::int32_t>);
                default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_input_element_type(1); return {};
            }
        case ngraph::element::Type_t::i16 :
            switch (node.get_input_element_type(1)) {
                case ngraph::element::Type_t::u8  : return make(ngraph::runtime::reference::cumsum<std::int16_t, std::uint8_t>);
                case ngraph::element::Type_t::i16 : return make(ngraph::runtime::reference::cumsum<std::int16_t, std::int16_t>);
                case ngraph::element::Type_t::u16 : return make(ngraph::runtime::reference::cumsum<std::int16_t, std::uint16_t>);
                case ngraph::element::Type_t::i32 : return make(ngraph::runtime::reference::cumsum<std::int16_t, std::int32_t>);
                default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_input_element_type(1); return {};
            }
        case ngraph::element::Type_t::u16 :
            switch (node.get_input_element_type(1)) {
                case ngraph::element::Type_t::u8  : return make(ngraph::runtime::reference::cumsum<std::uint16_t, std::uint8_t>);
                case ngraph::element::Type_t::i16 : return make(ngraph::runtime::reference::cumsum<std::uint16_t, std::int16_t>);
                case ngraph::element::Type_t::u16 : return make(ngraph::runtime::reference::cumsum<std::uint16_t, std::uint16_t>);
                case ngraph::element::Type_t::i32 : return make(ngraph::runtime::reference::cumsum<std::uint16_t, std::int32_t>);
                default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_input_element_type(1); return {};
            }
        case ngraph::element::Type_t::u32 :
            switch (node.get_input_element_type(1)) {
                case ngraph::element::Type_t::u8  : return make(ngraph::runtime::reference::cumsum<std::uint32_t, std::uint8_t>);
                case ngraph::element::Type_t::i16 : return make(ngraph::runtime::reference::cumsum<std::uint32_t, std::int16_t>);
                case ngraph::element::Type_t::u16 : return make(ngraph::runtime::reference::cumsum<std::uint32_t, std::uint16_t>);
                case ngraph::element::Type_t::i32 : return make(ngraph::runtime::reference::cumsum<std::uint32_t, std::int32_t>);
                default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_input_element_type(1); return {};
            }
        case ngraph::element::Type_t::i32 :
            switch (node.get_input_element_type(1)) {
                case ngraph::element::Type_t::u8  : return make(ngraph::runtime::reference::cumsum<std::int32_t, std::uint8_t>);
                case ngraph::element::Type_t::i16 : return make(ngraph::runtime::reference::cumsum<std::int32_t, std::int16_t>);
                case ngraph::element::Type_t::u16 : return make(ngraph::runtime::reference::cumsum<std::int32_t, std::uint16_t>);
                case ngraph::element::Type_t::i32 : return make(ngraph::runtime::reference::cumsum<std::int32_t, std::int32_t>);
                default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_input_element_type(1); return {};
            }
        case ngraph::element::Type_t::f32 :
            switch (node.get_input_element_type(1)) {
                case ngraph::element::Type_t::u8  : return make(ngraph::runtime::reference::cumsum<float, std::uint8_t>);
                case ngraph::element::Type_t::i16 : return make(ngraph::runtime::reference::cumsum<float, std::int16_t>);
                case ngraph::element::Type_t::u16 : return make(ngraph::runtime::reference::cumsum<float, std::uint16_t>);
                case ngraph::element::Type_t::i32 : return make(ngraph::runtime::reference::cumsum<float, int32_t>);
                default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_input_element_type(1); return {};
            }
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}

}  //  namespace ArmPlugin
