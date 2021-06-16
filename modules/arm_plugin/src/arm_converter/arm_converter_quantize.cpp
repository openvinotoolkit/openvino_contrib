// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "opset/quantize.hpp"
#include <arm_compute/runtime/NEON/functions/NEQuantizationLayer.h>
#include <arm_compute/runtime/NEON/functions/NEDequantizationLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmQuantize& node) {
    return MakeConversion<arm_compute::NEQuantizationLayer>(node.input(0), node.output(0));
}
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmDequantize& node) {
    return MakeConversion<arm_compute::NEDequantizationLayer>(node.input(0), node.output(0));
}
}