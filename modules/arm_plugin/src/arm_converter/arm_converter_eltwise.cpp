// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include <arm_compute/runtime/NEON/functions/NEArithmeticAddition.h>
#include <arm_compute/runtime/NEON/functions/NEArithmeticSubtraction.h>
#include <arm_compute/runtime/NEON/functions/NEElementwiseOperations.h>
#include <arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h>
#include <ngraph/runtime/reference/floor_mod.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Add& node) {
    return MakeConversion<arm_compute::NEArithmeticAddition>(node.input(0), node.input(1), node.output(0), arm_compute::ConvertPolicy::SATURATE);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Subtract& node) {
    return MakeConversion<arm_compute::NEArithmeticSubtraction>(node.input(0), node.input(1), node.output(0), arm_compute::ConvertPolicy::SATURATE);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Multiply& node) {
    return MakeConversion<arm_compute::NEPixelWiseMultiplication>(node.input(0), node.input(1), node.output(0),
        1.0f,
        arm_compute::ConvertPolicy::SATURATE,
        arm_compute::RoundingPolicy::TO_ZERO);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Minimum& node) {
    return MakeConversion<arm_compute::NEElementwiseMin>(node.input(0), node.input(1), node.output(0));
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Maximum& node) {
    return MakeConversion<arm_compute::NEElementwiseMax>(node.input(0), node.input(1), node.output(0));
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::SquaredDifference& node) {
    return MakeConversion<arm_compute::NEElementwiseSquaredDiff>(node.input(0), node.input(1), node.output(0));
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Power& node) {
    return MakeConversion<arm_compute::NEElementwisePower>(node.input(0), node.input(1), node.output(0));
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::FloorMod& node) {
    auto make = [&] (auto refFunction) {
        return MakeConversion(refFunction, node.input(0), node.input(1), node.output(0),
                              node.get_input_shape(0), node.get_input_shape(1), node.get_autob());
    };
    switch (node.input(0).get_element_type()) {
        case ngraph::element::Type_t::u8  : return make(ngraph::runtime::reference::floor_mod<std::uint8_t>);
        case ngraph::element::Type_t::i16 : return make(ngraph::runtime::reference::floor_mod<std::int16_t>);
        case ngraph::element::Type_t::u16 : return make(ngraph::runtime::reference::floor_mod<std::uint16_t>);
        case ngraph::element::Type_t::i32 : return make(ngraph::runtime::reference::floor_mod<std::int32_t>);
        case ngraph::element::Type_t::i64 : return make(ngraph::runtime::reference::floor_mod<std::int64_t>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::floor_mod<float>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}
} // namespace ArmPlugin
