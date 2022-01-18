// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/round.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Round& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.output(0),
                                    ngraph::shape_size(node.get_input_shape(0)),
                                    node.get_mode());
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::round),
        node.input(0), floatTypes);
}

}  //  namespace ArmPlugin
