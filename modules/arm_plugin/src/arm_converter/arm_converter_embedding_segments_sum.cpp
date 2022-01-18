// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/embedding_segments_sum.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::EmbeddingSegmentsSum& node) {
    auto make = [&] (auto refFunction) {
        if (node.get_input_size() > 5) {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        node.input(2),
                                        node.input(4),
                                        node.input(5),
                                        node.output(0),
                                        node.get_input_shape(0),
                                        node.get_input_shape(1),
                                        node.get_output_shape(0));
        } else if (node.get_input_size() > 4) {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        node.input(2),
                                        node.input(4),
                                        nullptr,
                                        node.output(0),
                                        node.get_input_shape(0),
                                        node.get_input_shape(1),
                                        node.get_output_shape(0));
        } else {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        node.input(2),
                                        nullptr,
                                        nullptr,
                                        node.output(0),
                                        node.get_input_shape(0),
                                        node.get_input_shape(1),
                                        node.get_output_shape(0));
        }
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::embeddingSegmentsSum),
        node.input(0), allTypes,
        node.input(1), indexTypes);
}

}  //  namespace ArmPlugin
