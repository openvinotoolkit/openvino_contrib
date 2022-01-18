// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0



#include <arm_compute/runtime/NEON/functions/NEPadLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Pad& node) {
    auto pads_begin = node.get_pads_begin();
    auto pads_end   = node.get_pads_end();
    arm_compute::PaddingList padding;
    for (size_t i = 0; i < pads_begin.size(); i++) {
        auto dim = AxisCast(i, pads_begin.size());
        padding.emplace_back(pads_begin[dim], pads_end[dim]);
    }

    arm_compute::PaddingMode mode;
    switch (node.get_pad_mode()) {
        case ngraph::op::PadMode::CONSTANT:
            mode = arm_compute::PaddingMode::CONSTANT;
            break;
        case ngraph::op::PadMode::REFLECT:
            mode = arm_compute::PaddingMode::REFLECT;
            break;
        case ngraph::op::PadMode::SYMMETRIC:
            mode = arm_compute::PaddingMode::SYMMETRIC;
            break;
        default:
            IE_THROW() << "Unsupported pad mode: " << node.get_pad_mode();
    }

    if (mode == arm_compute::PaddingMode::SYMMETRIC && !std::all_of(pads_end.begin(), pads_end.end(), [](int i){return i == 0;})) {
        IE_THROW() << "Unsupported SYMMETRIC pad mode with a non-zero pads end";
    }

    float value = safe_cast<opset::Constant>(node.input_value(3).get_node())->cast_vector<float>()[0];
    auto constant_value = mode == arm_compute::PaddingMode::CONSTANT ?
                          arm_compute::PixelValue(value) :
                          arm_compute::PixelValue();

    return MakeConversion<arm_compute::NEPadLayer>(node.input(0), node.output(0), padding, constant_value, mode);
}
} // namespace ArmPlugin