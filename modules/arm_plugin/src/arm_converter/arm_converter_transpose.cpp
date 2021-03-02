// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEPermute.h>
#include <ngraph/runtime/reference/transpose.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmTranspose& node) {
    enum {Data, Order};
    auto&& inputOrder = std::dynamic_pointer_cast<ngraph::op::Constant>(
                    node.input_value(Order).get_node_shared_ptr())->cast_vector<size_t>();

    if (inputOrder.empty()) {
        inputOrder.resize(node.get_input_shape(0).size());
        std::iota(inputOrder.begin(), inputOrder.end(), 0);
        std::reverse(inputOrder.begin(), inputOrder.end());
    }
    arm_compute::PermutationVector order;
    const auto maxSupportedNumOfDimensions = (inputOrder.size() < 4) ? 3u : 4u;
    for (unsigned int i = 0; i < maxSupportedNumOfDimensions; ++i) {
        order.set(i, i);
    }
    for (size_t i = 0; i < inputOrder.size(); ++i) {
        order.set(i, AxisCast(inputOrder[AxisCast(i, inputOrder.size())], inputOrder.size()));
    }
    return MakeConversion<arm_compute::NEPermute>(node.input(0), node.output(0), order);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Transpose& node) {
    auto make = [&] (auto refFunction) {
        if (ngraph::shape_size(node.get_input_shape(1)) == 0) {
            return MakeConversion(refFunction,
                                  node.input(0),
                                  node.output(0),
                                  node.get_input_shape(0),
                                  nullptr);
        }
        return MakeConversion(refFunction,
                              node.input(0),
                              node.output(0),
                              node.get_input_shape(0),
                              node.input(1));
    };

    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::u8 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::transpose<std::uint8_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::transpose<std::uint8_t, std::int64_t>);
        case ngraph::element::Type_t::i16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::transpose<std::int16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::transpose<std::int16_t, std::int64_t>);
        case ngraph::element::Type_t::u16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::transpose<std::uint16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::transpose<std::uint16_t, std::int64_t>);
        case ngraph::element::Type_t::u32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::transpose<std::uint32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::transpose<std::uint32_t, std::int64_t>);
        case ngraph::element::Type_t::i32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::transpose<std::int32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::transpose<std::int32_t, std::int64_t>);
        case ngraph::element::Type_t::i64 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::transpose<std::int64_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::transpose<std::int64_t, std::int64_t>);
        case ngraph::element::Type_t::f16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::transpose<ngraph::float16, std::int32_t>);
            }
            return make(ngraph::runtime::reference::transpose<ngraph::float16, std::int64_t>);
        case ngraph::element::Type_t::f32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::transpose<float, std::int32_t>);
            }
            return make(ngraph::runtime::reference::transpose<float, std::int64_t>);
        default: IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}
} // namespace ArmPlugin
