// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/gather_nd.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::GatherND& node) {
    if (node.get_output_shape(0).size() > 5) {
        IE_THROW() << "GatherND node doesn't support " << node.get_output_shape(0) << " output shape.";
    }

    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0), node.input(1), node.output(0),
                                    node.get_input_shape(0), node.get_input_shape(1), node.get_output_shape(0),
                                    static_cast<int>(node.get_batch_dims()));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::gather_nd),
        node.input(0), allTypes,
        node.input(1), indexTypes);
}

}  //  namespace ArmPlugin
