// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include <arm_compute/runtime/NEON/functions/NEL2NormalizeLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::NormalizeL2& node) {
    auto&& axes = node.get_reduction_axes();
    float eps   = node.get_eps();

    if (node.get_eps_mode() == ngraph::op::EpsMode::MAX) {
        THROW_IE_EXCEPTION << "Unsupported EpsMode::MAX of NormalizeL2 layer";
    }
    int axis = AxisCast(*axes.begin(), node.get_shape().size());
    //  Maximum supported actual reduction axis : 2
    if (axes.size() != 1 || axis > 2) {
        THROW_IE_EXCEPTION << "Unsupported NormalizeL2 layer with axes: " << axes;
    }
    return MakeConversion<arm_compute::NEL2NormalizeLayer>(node.input(0), node.output(0), axis, eps);
}
} //  namespace ArmPlugin
