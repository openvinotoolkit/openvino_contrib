// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEBatchToSpaceLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::BatchToSpace& node) {
    auto block_shape = safe_cast<opset::Constant>(node.input_value(1).get_node_shared_ptr());
    auto crops_begin = safe_cast<opset::Constant>(node.input_value(2).get_node_shared_ptr());
    auto crops_end   = safe_cast<opset::Constant>(node.input_value(3).get_node_shared_ptr());

    if (node.get_input_shape(0).size() != 4) {
        IE_THROW() << "Unsupported BatchToSpace with num dimensions != 4";
    }

    auto begin  = crops_begin->cast_vector<int>();
    auto end    = crops_end->cast_vector<int>();
    auto shapes = block_shape->cast_vector<int32_t>();

    if (!std::all_of(begin.begin(), begin.end(), [] (int i) {return i == 0;}) ||
        !std::all_of(end.begin(), end.end(), [] (int i) {return i == 0;})) {
        IE_THROW() << "Unsupported BatchToSpace op with crop > 0";
    }

    if (shapes[0] != 1 || shapes[1] != 1) {
        IE_THROW() << "Unsupported BatchToSpace op with block_shape != 1 for N, C";
    }
    int32_t block_shape_y = shapes[2];
    int32_t block_shape_x = shapes[3];
    return MakeConversion<arm_compute::NEBatchToSpaceLayer>(node.input(0), block_shape_x, block_shape_y, node.output(0));
}
} //  namespace ArmPlugin
