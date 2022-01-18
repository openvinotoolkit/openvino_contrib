// Copyright (C) 2020-2022 Intel Corporation
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
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::region_yolo),
        node.input(0), floatTypes);
}
}  //  namespace ArmPlugin
