// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <arm_compute/runtime/NEON/functions/NECopy.h>
#include <arm_compute/runtime/NEON/functions/NEConcatenateLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::Concat& node) {
    if (node.get_input_size() == 1) {
        return MakeConversion<arm_compute::NECopy>(node.input(0), node.output(0));
    }

    return MakeConversion<arm_compute::NEConcatenateLayer>(node.inputs(), node.output(0),
        AxisCast(node.get_axis(), node.get_input_shape(0).size()));
}
} // namespace ArmPlugin