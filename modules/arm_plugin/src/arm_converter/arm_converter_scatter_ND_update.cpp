// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/scatter_nd_update.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ScatterNDUpdate& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                              node.input(0),
                              node.input(1),
                              node.input(2),
                              node.output(0),
                              node.get_input_shape(0),
                              node.get_input_shape(1),
                              node.get_input_shape(2));
    };
    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::u8 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::scatterNdUpdate<std::uint8_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::scatterNdUpdate<std::uint8_t, std::int64_t>);
        case ngraph::element::Type_t::i16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::scatterNdUpdate<std::int16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::scatterNdUpdate<std::int16_t, std::int64_t>);
        case ngraph::element::Type_t::u16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::scatterNdUpdate<std::uint16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::scatterNdUpdate<std::uint16_t, std::int64_t>);
        case ngraph::element::Type_t::u32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::scatterNdUpdate<std::uint32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::scatterNdUpdate<std::uint32_t, std::int64_t>);
        case ngraph::element::Type_t::i32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::scatterNdUpdate<std::int32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::scatterNdUpdate<std::int32_t, std::int64_t>);
        case ngraph::element::Type_t::i64 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::scatterNdUpdate<std::int64_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::scatterNdUpdate<std::int64_t, std::int64_t>);
        case ngraph::element::Type_t::f16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::scatterNdUpdate<half_float::half, std::int32_t>);
            }
            return make(ngraph::runtime::reference::scatterNdUpdate<half_float::half, std::int64_t>);
        case ngraph::element::Type_t::f32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::scatterNdUpdate<float, std::int32_t>);
            }
            return make(ngraph::runtime::reference::scatterNdUpdate<float, std::int64_t>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

}  //  namespace ArmPlugin
