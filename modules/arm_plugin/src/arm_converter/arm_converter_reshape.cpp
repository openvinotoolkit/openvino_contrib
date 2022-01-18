// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEReshapeLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Reshape& node) {
    return MakeConversion<arm_compute::NEReshapeLayer>(node.input(0), node.output(0));}
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Squeeze& node) {
    return MakeConversion<arm_compute::NEReshapeLayer>(node.input(0), node.output(0));}
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Unsqueeze& node) {
    return MakeConversion<arm_compute::NEReshapeLayer>(node.input(0), node.output(0));}
} // namespace ArmPlugin
