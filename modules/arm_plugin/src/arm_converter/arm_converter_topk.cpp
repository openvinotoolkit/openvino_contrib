// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/topk.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::TopK& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.output(1),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_output_shape(0),
                                    static_cast<size_t>(node.get_axis()),
                                    node.get_k(),
                                    node.get_mode() == ngraph::op::TopKMode::MAX,
                                    node.get_sort_type());
    };

    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::u8 :
            if (node.get_index_element_type() == ngraph::element::i32) {
                return make(ngraph::runtime::reference::topk<std::uint8_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::topk<std::uint8_t, std::int64_t>);
        case ngraph::element::Type_t::i16 :
            if (node.get_index_element_type() == ngraph::element::i32) {
                return make(ngraph::runtime::reference::topk<std::int16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::topk<std::int16_t, std::int64_t>);
        case ngraph::element::Type_t::u16 :
            if (node.get_index_element_type() == ngraph::element::i32) {
                return make(ngraph::runtime::reference::topk<std::uint16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::topk<std::uint16_t, std::int64_t>);
        case ngraph::element::Type_t::u32 :
            if (node.get_index_element_type() == ngraph::element::i32) {
                return make(ngraph::runtime::reference::topk<std::uint32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::topk<std::uint32_t, std::int64_t>);
        case ngraph::element::Type_t::i32 :
            if (node.get_index_element_type() == ngraph::element::i32) {
                return make(ngraph::runtime::reference::topk<std::int32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::topk<std::int32_t, std::int64_t>);
        case ngraph::element::Type_t::i64 :
            if (node.get_index_element_type() == ngraph::element::i32) {
                return make(ngraph::runtime::reference::topk<std::int64_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::topk<std::int64_t, std::int64_t>);
        case ngraph::element::Type_t::f16 :
            if (node.get_index_element_type() == ngraph::element::i32) {
                return make(ngraph::runtime::reference::topk<ngraph::float16, std::int32_t>);
            }
            return make(ngraph::runtime::reference::topk<ngraph::float16, std::int64_t>);
        case ngraph::element::Type_t::f32 :
            if (node.get_index_element_type() == ngraph::element::i32) {
                return make(ngraph::runtime::reference::topk<float, std::int32_t>);
            }
            return make(ngraph::runtime::reference::topk<float, std::int64_t>);
        default: IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}

}  //  namespace ArmPlugin
