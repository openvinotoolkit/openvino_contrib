// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0



#include <arm_compute/runtime/NEON/functions/NEGather.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template <> Converter::Conversion::Ptr Converter::Convert(const opset::Gather& node) {
    auto axes = std::dynamic_pointer_cast<opset::Constant>(node.input_value(2).get_node_shared_ptr());
    if (!axes) {
        IE_THROW() << "Supported Gather op with constant axis only";
    }

    if (node.get_input_shape(1).size() > 1) {
        IE_THROW() << "Supported Gather op with scalar or 1D indices only";
    }

    int axis = axes->cast_vector<int64_t>()[0];
    if (axis < 0) {
        axis += node.get_input_shape(0).size();
    }
    axis = AxisCast(axis, node.get_input_shape(0).size());
    return MakeConversion<arm_compute::NEGather>(node.input(0), node.input(1), node.output(0), axis);
}
}  //  namespace ArmPlugin
