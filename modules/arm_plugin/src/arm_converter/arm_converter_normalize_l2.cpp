// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEL2NormalizeLayer.h>
#include <ngraph/runtime/reference/normalize_l2.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::NormalizeL2& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_reduction_axes(),
                                    node.get_eps(),
                                    node.get_eps_mode());
    };
    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::normalize_l2<ngraph::float16>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::normalize_l2<float>);
        default: IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmNormalizeL2& node) {
    auto&& axes = node.get_reduction_axes();
    float eps   = node.get_eps();

    if (node.get_eps_mode() == ngraph::op::EpsMode::ADD) {
        IE_THROW() << "Unsupported EpsMode::ADD of NormalizeL2 layer. Use decomposition transform.";
    }
    int axis = AxisCast(*axes.begin(), node.get_shape().size());
    //  Maximum supported actual reduction axis : 2
    if (axes.size() != 1 || axis > 2) {
        IE_THROW() << "Unsupported NormalizeL2 layer with axes: " << axes;
    }
    return MakeConversion<arm_compute::NEL2NormalizeLayer>(node.input(0), node.output(0), axis, eps);
}
} //  namespace ArmPlugin
