// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEPermute.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Transpose& node) {
    enum {Data, Order};
    auto&& inputOrder = std::dynamic_pointer_cast<ngraph::op::Constant>(
                    node.input_value(Order).get_node_shared_ptr())->cast_vector<size_t>();
    arm_compute::PermutationVector order;
    const auto maxSupportedNumOfDimensions = (inputOrder.size() < 4) ? 3u : 4u;
    for (unsigned int i = 0; i < maxSupportedNumOfDimensions; ++i) {
        order.set(i, i);
    }
    for (size_t i = 0; i < inputOrder.size(); ++i) {
        order.set(i, AxisCast(inputOrder[AxisCast(i, inputOrder.size())], inputOrder.size()));
    }
    return MakeConversion<arm_compute::NEPermute>(node.input(0), node.output(0), order);
}
} // namespace ArmPlugin
