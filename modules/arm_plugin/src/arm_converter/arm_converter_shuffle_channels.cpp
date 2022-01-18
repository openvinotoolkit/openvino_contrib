// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEActivationLayer.h>
#include <arm_compute/runtime/NEON/functions/NEChannelShuffleLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ShuffleChannels& node) {
    int axis = node.get_axis();
    if (axis < 0) {
        axis += node.get_input_shape(0).size();
    }

    unsigned int group = static_cast<unsigned int>(node.get_group());
    if (axis != 1) {
        IE_THROW() << "Unsupported axis: " << axis;
    }
    if (group == 1) {
        arm_compute::ActivationLayerInfo info(arm_compute::ActivationLayerInfo::ActivationFunction::IDENTITY);
        return MakeConversion<arm_compute::NEActivationLayer>(node.input(0), node.output(0), info);
    }
    return MakeConversion<arm_compute::NEChannelShuffleLayer>(node.input(0), node.output(0), group);
}
} //  namespace ArmPlugin
