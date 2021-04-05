// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/reverse_sequence.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReverseSequence& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_batch_axis(),
                                    node.get_sequence_axis(),
                                    node.input(1));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::reverse_sequence),
        node.input(0), allTypes,
        node.input(1), indexTypes);
}

}  //  namespace ArmPlugin
