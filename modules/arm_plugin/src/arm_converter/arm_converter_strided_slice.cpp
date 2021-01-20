// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <details/ie_exception.hpp>

#include <arm_compute/runtime/NEON/functions/NEStridedSlice.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template <> Converter::Conversion::Ptr Converter::Convert(const opset::StridedSlice& node) {
    auto&& begin  = std::dynamic_pointer_cast<ngraph::op::Constant>(
                        node.input(1).get_source_output().get_node_shared_ptr())->cast_vector<int>();
    auto&& end    = std::dynamic_pointer_cast<ngraph::op::Constant>(
                        node.input(2).get_source_output().get_node_shared_ptr())->cast_vector<int>();
    auto&& stride = std::dynamic_pointer_cast<ngraph::op::Constant>(
                        node.input(3).get_source_output().get_node_shared_ptr())->cast_vector<int>();

    arm_compute::Coordinates starts, finishes, deltas;
    for (size_t i = 0; i < begin.size(); ++i) {
        auto axis = AxisCast(i, begin.size());
        starts.set(axis, begin[i]);
        finishes.set(axis, end[i]);
        deltas.set(axis, stride[i]);
    }

    return MakeConversion<arm_compute::NEStridedSlice>(node.input(0), node.output(0), starts, finishes, deltas);
}
}  //  namespace ArmPlugin