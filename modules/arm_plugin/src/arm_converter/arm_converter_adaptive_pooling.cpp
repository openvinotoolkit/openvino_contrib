// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <ngraph/runtime/reference/adaptive_avg_pool.hpp>
#include <ngraph/runtime/reference/adaptive_max_pool.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::v8::AdaptiveAvgPool& node) {
    auto make = [&] (auto refFunction) {
    return this->MakeConversion(refFunction,
                                node.input(0),
                                node.output(0),
                                node.get_input_shape(0),
                                node.get_output_shape(0));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::adaptive_avg_pool),
        node.input(0), floatTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::v8::AdaptiveMaxPool& node) {
    auto make = [&] (auto refFunction) {
    return this->MakeConversion(refFunction,
                                node.input(0),
                                node.output(0),
                                node.output(1),
                                node.get_input_shape(0),
                                node.get_output_shape(0));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::adaptive_max_pool),
        node.input(0), floatTypes,
        node.output(1), indexTypes);
}

} // namespace ArmPlugin