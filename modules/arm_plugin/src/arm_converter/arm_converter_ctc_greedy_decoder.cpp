// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/ctc_greedy_decoder.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::CTCGreedyDecoder& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_output_shape(0),
                                    node.get_ctc_merge_repeated());
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::ctc_greedy_decoder),
        node.input(0), floatTypes);
}

}  //  namespace ArmPlugin
