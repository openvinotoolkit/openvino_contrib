// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/gather_nd.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::GatherND& node) {
    if (node.get_output_shape(0).size() > 5) {
        THROW_IE_EXCEPTION << "GatherND node doesn't support " << node.get_output_shape(0) << " output shape.";
    }

    auto make = [&] (auto refFunction) {
        return MakeConversion(refFunction,
                              node.input(0), node.input(1), node.output(0),
                              node.get_input_shape(0), node.get_input_shape(1), node.get_output_shape(0),
                              static_cast<int>(node.get_batch_dims()));
    };
    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::u8 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_nd<std::uint8_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_nd<std::uint8_t, std::int64_t>);
        case ngraph::element::Type_t::i16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_nd<std::int16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_nd<std::int16_t, std::int64_t>);
        case ngraph::element::Type_t::u16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_nd<std::uint16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_nd<std::uint16_t, std::int64_t>);
        case ngraph::element::Type_t::u32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_nd<std::uint32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_nd<std::uint32_t, std::int64_t>);
        case ngraph::element::Type_t::i32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_nd<std::int32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_nd<std::int32_t, std::int64_t>);
        case ngraph::element::Type_t::i64 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_nd<std::int64_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_nd<std::int64_t, std::int64_t>);
        case ngraph::element::Type_t::f16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_nd<half_float::half, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_nd<half_float::half, std::int64_t>);
        case ngraph::element::Type_t::f32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather_nd<float, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather_nd<float, std::int64_t>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

}  //  namespace ArmPlugin
