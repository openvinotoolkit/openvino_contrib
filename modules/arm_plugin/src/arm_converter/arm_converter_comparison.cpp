// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEElementwiseOperations.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<typename Comparison>
static auto ConvertComparison(const Comparison& node, const arm_compute::ComparisonOperation& op, Converter* converter) {
    return converter->MakeConversion<arm_compute::NEElementwiseComparison>(node.input(0), node.input(1), node.output(0), op);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Equal& node) {
    return ConvertComparison(node, arm_compute::ComparisonOperation::Equal, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::NotEqual& node) {
    return ConvertComparison(node, arm_compute::ComparisonOperation::NotEqual, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Greater& node) {
    return ConvertComparison(node, arm_compute::ComparisonOperation::Greater, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::GreaterEqual& node) {
    return ConvertComparison(node, arm_compute::ComparisonOperation::GreaterEqual, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Less& node) {
    return ConvertComparison(node, arm_compute::ComparisonOperation::Less, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::LessEqual& node) {
    return ConvertComparison(node, arm_compute::ComparisonOperation::LessEqual, this);
}
} // namespace ArmPlugin
