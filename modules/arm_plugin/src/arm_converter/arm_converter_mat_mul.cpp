// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <details/ie_exception.hpp>

#include <arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
enum Input {Features, Weights, Bias};
template<> Converter::Conversion::Ptr Converter::Convert(const opset::MatMul& node) {
    if (node.get_transpose_a()) {
        THROW_IE_EXCEPTION << "Can not create MatMul layer with transpose first input";
    }
    return MakeConversion<arm_compute::NEFullyConnectedLayer>(node.input(Features), node.input(Weights), nullptr, node.output(0));
}
template<> Converter::Conversion::Ptr Converter::Convert(const opset::MatMulBias& node) {
    if (node.get_transpose_a()) {
        THROW_IE_EXCEPTION << "Can not create MatMul layer with transpose first input";
    }
    return MakeConversion<arm_compute::NEFullyConnectedLayer>(node.input(Features), node.input(Weights), node.input(Bias), node.output(0));
}
}  //  namespace ArmPlugin
