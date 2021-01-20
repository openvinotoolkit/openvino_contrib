// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/region_yolo.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::RegionYolo& node) {
    auto make = [&] (auto refFunction) {
        return MakeConversion(refFunction,
                              node.input(0),
                              node.output(0),
                              node.get_input_shape(0),
                              static_cast<int>(node.get_num_coords()),
                              static_cast<int>(node.get_num_classes()),
                              static_cast<int>(node.get_num_regions()),
                              node.get_do_softmax(),
                              node.get_mask());
    };

    if (node.input(0).get_element_type() != ngraph::element::f32) {
        THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type();
    }
    return make(ngraph::runtime::reference::region_yolo<float>);
}
}  //  namespace ArmPlugin
