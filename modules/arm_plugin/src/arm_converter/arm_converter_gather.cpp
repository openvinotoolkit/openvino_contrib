// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0



#include <arm_compute/runtime/NEON/functions/NEGather.h>
#include <ngraph/runtime/reference/gather.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template <> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::v1::Gather& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_output_shape(0),
                                    static_cast<size_t>(node.get_axis()));
    };
    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::u8 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather<std::uint8_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather<std::uint8_t, std::int64_t>);
        case ngraph::element::Type_t::i16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather<std::int16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather<std::int16_t, std::int64_t>);
        case ngraph::element::Type_t::u16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather<std::uint16_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather<std::uint16_t, std::int64_t>);
        case ngraph::element::Type_t::u32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather<std::uint32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather<std::uint32_t, std::int64_t>);
        case ngraph::element::Type_t::i32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather<std::int32_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather<std::int32_t, std::int64_t>);
        case ngraph::element::Type_t::i64 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather<std::int64_t, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather<std::int64_t, std::int64_t>);
        case ngraph::element::Type_t::f16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather<ngraph::float16, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather<ngraph::float16, std::int64_t>);
        case ngraph::element::Type_t::f32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::gather<float, std::int32_t>);
            }
            return make(ngraph::runtime::reference::gather<float, std::int64_t>);
        default: IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}

template <> Converter::Conversion::Ptr Converter::Convert(const opset::ArmGather& node) {
    auto axes = std::dynamic_pointer_cast<opset::Constant>(node.input_value(2).get_node_shared_ptr());
    if (!axes) {
        IE_THROW() << "Supported Gather op with constant axis only";
    }

    if (node.get_input_shape(1).size() > 1) {
        IE_THROW() << "Supported Gather op with scalar or 1D indices only";
    }

    int axis = axes->cast_vector<int64_t>()[0];
    if (axis < 0) {
        axis += node.get_input_shape(0).size();
    }
    axis = AxisCast(axis, node.get_input_shape(0).size());
    return MakeConversion<arm_compute::NEGather>(node.input(0), node.input(1), node.output(0), axis);
}
}  //  namespace ArmPlugin
