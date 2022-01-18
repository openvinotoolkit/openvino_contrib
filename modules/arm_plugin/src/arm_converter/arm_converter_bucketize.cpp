// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <ngraph/runtime/reference/bucketize.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Bucketize& node) {
    auto make = [&] (auto refFunction) {
    return this->MakeConversion(refFunction,
                                node.input(0),
                                node.input(1),
                                node.output(0),
                                node.get_input_shape(0),
                                node.get_input_shape(1),
                                node.get_with_right_bound());
    };
    auto numericTypes = std::tuple<std::int32_t, std::int64_t, ngraph::float16, float>{};
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::bucketize),
        node.get_input_element_type(0), numericTypes,
        node.get_input_element_type(1), numericTypes,
        node.get_output_type(), indexTypes);
}

} // namespace ArmPlugin