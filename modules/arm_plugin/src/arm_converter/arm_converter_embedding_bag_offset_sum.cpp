// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/embedding_bag_offsets_sum.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::EmbeddingBagOffsetsSum& node) {
    auto make = [&] (auto refFunction) {
        if (node.get_input_size() > 4) {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        node.input(2),
                                        node.input(3),
                                        node.input(4),
                                        node.output(0),
                                        ngraph::shape_size(node.get_input_shape(1)),
                                        node.get_shape());
        } else if (node.get_input_size() > 3) {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        node.input(2),
                                        node.input(3),
                                        nullptr,
                                        node.output(0),
                                        ngraph::shape_size(node.get_input_shape(1)),
                                        node.get_shape());
        } else {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        node.input(2),
                                        nullptr,
                                        nullptr,
                                        node.output(0),
                                        ngraph::shape_size(node.get_input_shape(1)),
                                        node.get_shape());
        }
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::embeddingBagOffsetsSum),
        node.get_input_element_type(0), allTypes,
        node.get_input_element_type(1), indexTypes);
}

}  //  namespace ArmPlugin
