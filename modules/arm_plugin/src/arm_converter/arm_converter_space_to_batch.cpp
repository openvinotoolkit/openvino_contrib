// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NESpaceToBatchLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::SpaceToBatch& node) {
    auto block_shape = safe_cast<opset::Constant>(node.input_value(1).get_node_shared_ptr());
    auto pads_begin = safe_cast<opset::Constant>(node.input_value(2).get_node_shared_ptr());
    auto pads_end   = safe_cast<opset::Constant>(node.input_value(3).get_node_shared_ptr());

    if (node.get_input_shape(0).size() != 4) {
        IE_THROW() << "Unsupported SpaceToBatch with num dimensions != 4";
    }

    auto begin  = pads_begin->cast_vector<int>();
    auto end    = pads_end->cast_vector<int>();
    auto shapes = block_shape->cast_vector<int32_t>();

    if (begin[0] != 0 || begin[1] != 0 || end[0] != 0 || end[1] != 0) {
        IE_THROW() << "Unsupported SpaceToBatch op with non-zero paddings for batch or channels";
    }

    if (shapes[0] != 1 || shapes[1] != 1) {
        IE_THROW() << "Unsupported SpaceToBatch op with shapes != 1 for batch or channels";
    }
    int32_t block_shape_y = shapes[2];
    int32_t block_shape_x = shapes[3];

    arm_compute::Size2D padding_left(begin[3], begin[2]);
    arm_compute::Size2D padding_right(end[3], end[2]);
    return MakeConversion<arm_compute::NESpaceToBatchLayer>(node.input(0), block_shape_x, block_shape_y, padding_left, padding_right, node.output(0));
}
} //  namespace ArmPlugin
