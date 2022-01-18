// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/topk.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::TopK& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.output(1),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_output_shape(0),
                                    static_cast<size_t>(node.get_axis()),
                                    node.get_k(),
                                    node.get_mode() == ngraph::op::TopKMode::MAX,
                                    node.get_sort_type());
    };

    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::topk),
        node.input(0), allTypes,
        node.get_index_element_type(),  indexTypes);
}

}  //  namespace ArmPlugin
