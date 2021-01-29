// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NESplit.h>
#include <arm_compute/runtime/NEON/functions/NECopy.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Split& node) {
    size_t numDimensions = node.get_output_shape(0).size();
    int axis = std::dynamic_pointer_cast<ngraph::op::Constant>(
        node.input_value(1).get_node_shared_ptr())->cast_vector<int>()[0];
    if (axis < 0) {
        axis += numDimensions;
    }
    if (node.get_output_size() == 1) {
        return MakeConversion<arm_compute::NECopy>(node.input(0), node.output(0));
    }
    return MakeConversion<arm_compute::NESplit>(node.input(0), node.outputs(),
        static_cast<unsigned int>(AxisCast(axis, numDimensions)));
}
}  //  namespace ArmPlugin