// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/scatter_nd_update.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ScatterNDUpdate& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.input(2),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_input_shape(2));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::scatterNdUpdate),
        node.input(0), allTypes,
        node.input(1), indexTypes);
}

}  //  namespace ArmPlugin
