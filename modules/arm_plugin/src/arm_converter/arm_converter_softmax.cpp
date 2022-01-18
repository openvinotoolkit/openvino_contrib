// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <arm_compute/runtime/NEON/functions/NESoftmaxLayer.h>
#include <ngraph/runtime/reference/softmax.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Softmax& node) {
    if (true) {
        return MakeConversion<arm_compute::NESoftmaxLayer>(node.input(0),
                                                        node.output(0),
                                                        1.0f,
                                                        static_cast<int32_t>(AxisCast(node.get_axis(), node.get_shape().size())));
    } else {
        auto make = [&] (auto refFunction) {
            return this->MakeConversion(refFunction, node.input(0), node.output(0), node.get_shape(), ngraph::AxisSet{node.get_axis()});
        };
        return CallSwitch(
            AP_WRAP(make, ngraph::runtime::reference::softmax),
            node.input(0), floatTypes);
    }
}
} // namespace ArmPlugin
