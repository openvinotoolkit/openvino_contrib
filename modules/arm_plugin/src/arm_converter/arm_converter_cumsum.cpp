// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/cum_sum.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::CumSum& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.input(1), node.output(0),
                                    node.get_input_shape(0), node.is_exclusive(), node.is_reverse());
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::cumsum),
        node.input(0), allTypes,
        node.input(1), intTypes);
}
}  //  namespace ArmPlugin
