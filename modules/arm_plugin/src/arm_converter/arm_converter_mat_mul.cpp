// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
enum InputArg {Features, Weights, Bias};
template<> Converter::Conversion::Ptr Converter::Convert(const opset::MatMul& node) {
    if (node.get_transpose_a()) {
        IE_THROW() << "Can not create MatMul layer with transpose first input";
    }
    return MakeConversion<arm_compute::NEFullyConnectedLayer>(node.input(Features), node.input(Weights), nullptr, node.output(0));
}
template<> Converter::Conversion::Ptr Converter::Convert(const opset::MatMulBias& node) {
    if (node.get_transpose_a()) {
        IE_THROW() << "Can not create MatMul layer with transpose first input";
    }
    return MakeConversion<arm_compute::NEFullyConnectedLayer>(node.input(Features), node.input(Weights), node.input(Bias), node.output(0));
}
}  //  namespace ArmPlugin
