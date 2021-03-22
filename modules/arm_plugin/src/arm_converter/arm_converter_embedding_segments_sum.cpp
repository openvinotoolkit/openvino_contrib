// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/embedding_segments_sum.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::EmbeddingSegmentsSum& node) {
    auto make = [&] (auto refFunction) {
        if (node.get_input_size() > 5) {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        node.input(2),
                                        node.input(4),
                                        node.input(5),
                                        node.output(0),
                                        node.get_input_shape(0),
                                        node.get_input_shape(1),
                                        node.get_output_shape(0));
        } else if (node.get_input_size() > 4) {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        node.input(2),
                                        node.input(4),
                                        nullptr,
                                        node.output(0),
                                        node.get_input_shape(0),
                                        node.get_input_shape(1),
                                        node.get_output_shape(0));
        } else {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        node.input(2),
                                        nullptr,
                                        nullptr,
                                        node.output(0),
                                        node.get_input_shape(0),
                                        node.get_input_shape(1),
                                        node.get_output_shape(0));
        }
    };

    ngraph::element::Type_t indicesType = node.get_input_element_type(1);
    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::u8 :
            if (indicesType == ngraph::element::i32) {
                return make(ngraph::runtime::reference::embeddingSegmentsSum<std::uint8_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingSegmentsSum<std::uint8_t, std::int64_t>);
        case ngraph::element::Type_t::i16 :
            if (indicesType == ngraph::element::i32) {
                return make(ngraph::runtime::reference::embeddingSegmentsSum<std::int16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingSegmentsSum<std::int16_t, std::int64_t>);
        case ngraph::element::Type_t::u16 :
            if (indicesType == ngraph::element::i32) {
                return make(ngraph::runtime::reference::embeddingSegmentsSum<std::uint16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingSegmentsSum<std::uint16_t, std::int64_t>);
        case ngraph::element::Type_t::u32 :
            if (indicesType == ngraph::element::i32) {
                return make(ngraph::runtime::reference::embeddingSegmentsSum<std::uint32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingSegmentsSum<std::uint32_t, std::int64_t>);
        case ngraph::element::Type_t::i32 :
            if (indicesType == ngraph::element::i32) {
                return make(ngraph::runtime::reference::embeddingSegmentsSum<std::int32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingSegmentsSum<std::int32_t, std::int64_t>);
        case ngraph::element::Type_t::i64 :
            if (indicesType == ngraph::element::i32) {
                return make(ngraph::runtime::reference::embeddingSegmentsSum<std::int64_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingSegmentsSum<std::int64_t, std::int64_t>);
        case ngraph::element::Type_t::f16 :
            if (indicesType == ngraph::element::i32) {
                return make(ngraph::runtime::reference::embeddingSegmentsSum<half_float::half, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingSegmentsSum<half_float::half, std::int64_t>);
        case ngraph::element::Type_t::f32 :
            if (indicesType == ngraph::element::i32) {
                return make(ngraph::runtime::reference::embeddingSegmentsSum<float, std::int32_t>);
            }
            return make(ngraph::runtime::reference::embeddingSegmentsSum<float, std::int64_t>);
        default: IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}

}  //  namespace ArmPlugin
