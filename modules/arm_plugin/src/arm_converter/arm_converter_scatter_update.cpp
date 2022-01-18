// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/scatter_update.hpp>

namespace ArmPlugin {
template <typename T, typename U>
void wrap_scatter_update(const T* input_data,
                         const U* indices,
                         const T* updates,
                         const U* axis,
                         T* out,
                         const size_t elem_size,
                         const ngraph::Shape& data_shape,
                         const ngraph::Shape& indices_shape,
                         const ngraph::Shape& updates_shape) {
    auto indices_size = ngraph::shape_size(indices_shape);
    std::vector<std::int64_t> converted_indices(indices_size);
    for (size_t i = 0; i < indices_size; i++) {
        converted_indices[i] = static_cast<std::int64_t>(indices[i]);
    }
    std::int64_t axis_val = static_cast<std::int64_t>(axis[0]);
    if (axis_val < 0) {
        axis_val = static_cast<std::int64_t>(data_shape.size()) + axis_val;
    }
    ngraph::runtime::reference::scatter_update(reinterpret_cast<const char*>(input_data),
                                               converted_indices.data(),
                                               reinterpret_cast<const char*>(updates),
                                               axis_val,
                                               reinterpret_cast<char*>(out),
                                               elem_size,
                                               data_shape,
                                               indices_shape,
                                               updates_shape);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ScatterUpdate& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.input(2),
                                    node.input(3),
                                    node.output(0),
                                    node.get_input_element_type(0).size(),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_input_shape(2));
    };
    return CallSwitch(
        AP_WRAP(make, wrap_scatter_update),
        node.input(0), allTypes,
        node.input(1), indexTypes);
}

}  //  namespace ArmPlugin
