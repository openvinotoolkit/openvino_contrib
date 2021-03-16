// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/embedding_bag_offsets_sum.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::EmbeddingBagOffsetsSum& node) {
    auto make = [&] (auto refFunction) {
        if (node.get_input_size() > 4) {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        node.input(2),
                                        node.input(3),
                                        node.input(4),
                                        node.output(0),
                                        ngraph::shape_size(node.get_input_shape(1)),
                                        node.get_shape());
        } else if (node.get_input_size() > 3) {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        node.input(2),
                                        node.input(3),
                                        nullptr,
                                        node.output(0),
                                        ngraph::shape_size(node.get_input_shape(1)),
                                        node.get_shape());
        } else {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        node.input(2),
                                        nullptr,
                                        nullptr,
                                        node.output(0),
                                        ngraph::shape_size(node.get_input_shape(1)),
                                        node.get_shape());
        }
    };

    ngraph::element::Type_t indicesType = node.get_input_element_type(1);
    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::u8 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagOffsetsSum<std::uint8_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagOffsetsSum<std::uint8_t, size_t>);
        case ngraph::element::Type_t::i16 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagOffsetsSum<std::int16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagOffsetsSum<std::int16_t, size_t>);
        case ngraph::element::Type_t::u16 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagOffsetsSum<std::uint16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagOffsetsSum<std::uint16_t, size_t>);
        case ngraph::element::Type_t::u32 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagOffsetsSum<std::uint32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagOffsetsSum<std::uint32_t, size_t>);
        case ngraph::element::Type_t::i32 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagOffsetsSum<std::int32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagOffsetsSum<std::int32_t, size_t>);
        case ngraph::element::Type_t::i64 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagOffsetsSum<std::int64_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagOffsetsSum<std::int64_t, size_t>);
        case ngraph::element::Type_t::f16 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagOffsetsSum<half_float::half, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagOffsetsSum<half_float::half, size_t>);
        case ngraph::element::Type_t::f32 :
            if (indicesType == ngraph::element::i32 || indicesType == ngraph::element::u32) {
                return make(ngraph::runtime::reference::embeddingBagOffsetsSum<float, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingBagOffsetsSum<float, size_t>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

}  //  namespace ArmPlugin
