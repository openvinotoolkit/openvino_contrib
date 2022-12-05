// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <arm_compute/runtime/NEON/functions/NESoftmaxLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Softmax& node) {
    return MakeConversion<arm_compute::NESoftmaxLayer>(node.input(0),
                                                       node.output(0),
                                                       1.0f,
                                                       static_cast<int32_t>(AxisCast(node.get_axis(), node.get_shape().size())));
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::LogSoftmax& node) {
    auto axis = node.get_axis();
    if (axis < 0) { axis += node.get_shape().size(); }
    return MakeConversion<arm_compute::NELogSoftmaxLayer>(node.input(0),
                                                          node.output(0),
                                                          1.0f,
                                                          static_cast<int32_t>(AxisCast(axis, node.get_shape().size())));
}
} // namespace ArmPlugin
