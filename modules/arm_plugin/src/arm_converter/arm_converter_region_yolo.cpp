// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/region_yolo.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::RegionYolo& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    static_cast<int>(node.get_num_coords()),
                                    static_cast<int>(node.get_num_classes()),
                                    static_cast<int>(node.get_num_regions()),
                                    node.get_do_softmax(),
                                    node.get_mask());
    };

    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::region_yolo<ngraph::float16>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::region_yolo<float>);
        default: IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}
}  //  namespace ArmPlugin
