// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/ctc_loss.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::CTCLoss& node) {
    if (node.get_input_size() < 4) {
        IE_THROW() << "Unsupported CTCLoss op with num inputs = " << node.get_input_size();
    }

    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.get_input_shape(0),
                                    node.input(1),
                                    node.input(2),
                                    node.input(3),
                                    node.input(4),
                                    node.get_preprocess_collapse_repeated(),
                                    node.get_ctc_merge_repeated(),
                                    node.get_unique(),
                                    node.output(0));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::CTCLoss),
        node.input(0), floatTypes,
        node.input(1), indexTypes);
}

}  //  namespace ArmPlugin
