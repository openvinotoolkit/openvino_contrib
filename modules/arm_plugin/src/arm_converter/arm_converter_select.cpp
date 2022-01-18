// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0



#include <arm_compute/runtime/NEON/functions/NESelect.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template <> Converter::Conversion::Ptr Converter::Convert(const opset::Select& node) {
    if (node.get_input_shape(0) != node.get_input_shape(1) || node.get_input_shape(0) != node.get_input_shape(2)) {
        IE_THROW() << "Select op doesn't support broadcast";
    }
    return MakeConversion<arm_compute::NESelect>(node.input(0), node.input(1), node.input(2), node.output(0));
}
}  //  namespace ArmPlugin
