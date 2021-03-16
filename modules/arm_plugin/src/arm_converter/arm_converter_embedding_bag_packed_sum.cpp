// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/embedding_bag_packed_sum.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::EmbeddingBagPackedSum& node) {
    auto make = [&] (auto refFunction) {
        if (node.get_input_size() > 2) {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        node.input(2),
                                        node.output(0),
                                        node.get_input_shape(1),
                                        node.get_shape());
        } else {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        nullptr,
                                        node.output(0),
                                        node.get_input_shape(1),
                                        node.get_shape());
        }
    };
    ngraph::element::Type_t indicesType = node.get_input_element_type(1);
    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::u8 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagPackedSum<std::uint8_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagPackedSum<std::uint8_t, size_t>);
        case ngraph::element::Type_t::i16 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagPackedSum<std::int16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagPackedSum<std::int16_t, size_t>);
        case ngraph::element::Type_t::u16 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagPackedSum<std::uint16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagPackedSum<std::uint16_t, size_t>);
        case ngraph::element::Type_t::u32 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagPackedSum<std::uint32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagPackedSum<std::uint32_t, size_t>);
        case ngraph::element::Type_t::i32 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagPackedSum<std::int32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagPackedSum<std::int32_t, size_t>);
        case ngraph::element::Type_t::i64 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagPackedSum<std::int64_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagPackedSum<std::int64_t, size_t>);
        case ngraph::element::Type_t::f16 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagPackedSum<half_float::half, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagPackedSum<half_float::half, size_t>);
        case ngraph::element::Type_t::f32 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagPackedSum<float, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagPackedSum<float, size_t>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

}  //  namespace ArmPlugin
