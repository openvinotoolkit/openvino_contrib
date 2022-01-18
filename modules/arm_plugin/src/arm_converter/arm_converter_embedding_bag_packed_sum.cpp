// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/embedding_bag_packed_sum.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::EmbeddingBagPackedSum& node) {
    auto make = [&] (auto refFunction) {
        if (node.get_input_size() > 2) {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        node.input(2),
                                        node.output(0),
                                        node.get_input_shape(1),
                                        node.get_shape());
        } else {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        nullptr,
                                        node.output(0),
                                        node.get_input_shape(1),
                                        node.get_shape());
        }
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::embeddingBagPackedSum),
        node.input(0), allTypes,
        node.input(1), indexTypes);
}

}  //  namespace ArmPlugin
