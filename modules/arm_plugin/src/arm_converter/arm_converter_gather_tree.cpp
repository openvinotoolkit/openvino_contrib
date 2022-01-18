// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/gather_tree.hpp>

namespace ArmPlugin {
template <typename T>
void wrap_gather_tree(const T* step_ids,
                      const T* parent_ids,
                      const T* max_seq_len,
                      const T* end_token,
                      T* out,
                      const ngraph::Shape& step_ids_shape,
                      const ngraph::Shape& parent_ids_shape,
                      const ngraph::Shape& max_seq_len_shape,
                      const ngraph::Shape& end_token_shape,
                      const ngraph::element::Type& type) {
    ngraph::runtime::reference::gather_tree(reinterpret_cast<const char*>(step_ids),
                                            reinterpret_cast<const char*>(parent_ids),
                                            reinterpret_cast<const char*>(max_seq_len),
                                            reinterpret_cast<const char*>(end_token),
                                            reinterpret_cast<char*>(out),
                                            step_ids_shape,
                                            parent_ids_shape,
                                            max_seq_len_shape,
                                            end_token_shape,
                                            type);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::GatherTree& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.input(2),
                                    node.input(3),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_input_shape(2),
                                    node.get_input_shape(3),
                                    node.get_input_element_type(1));
    };
    return CallSwitch(
        AP_WRAP(make, wrap_gather_tree),
        node.input(0), allTypes);
}

}  //  namespace ArmPlugin
