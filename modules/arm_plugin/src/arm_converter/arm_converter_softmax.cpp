// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>
#include <arm_compute/runtime/NEON/functions/NESoftmaxLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Softmax& node) {
    return MakeConversion<arm_compute::NESoftmaxLayer>(node.input(0),
                                                       node.output(0),
                                                       1.0f,
                                                       static_cast<int32_t>(AxisCast(node.get_axis(), node.get_shape().size())));
}
} // namespace ArmPlugin
