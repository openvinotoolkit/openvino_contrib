// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/roi_pooling.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ROIPooling& node) {
    auto make = [&] (auto refFunction) {
        return MakeConversion(refFunction, node.input(0), node.input(1), node.output(0),
                              node.get_input_shape(0), node.get_input_shape(1), node.get_output_shape(0),
                              node.get_spatial_scale(), node.get_method());
    };

    if (node.input(0).get_element_type() != ngraph::element::f32) {
        THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type();
    }
    return make(ngraph::runtime::reference::roi_pooling<float>);
}

}  //  namespace ArmPlugin
