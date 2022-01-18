// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEPermute.h>
#include <ngraph/runtime/reference/transpose.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmTranspose& node) {
    enum {Data, Order};
    auto inputOrder = safe_cast<ngraph::op::Constant>(
                    node.input_value(Order).get_node_shared_ptr())->cast_vector<size_t>();

    if (inputOrder.empty()) {
        inputOrder.resize(node.get_input_shape(0).size());
        std::iota(inputOrder.begin(), inputOrder.end(), 0);
        std::reverse(inputOrder.begin(), inputOrder.end());
    }
    arm_compute::PermutationVector order;
    const auto maxSupportedNumOfDimensions = (inputOrder.size() < 4) ? 3u : 4u;
    for (unsigned int i = 0; i < maxSupportedNumOfDimensions; ++i) {
        order.set(i, i);
    }
    for (size_t i = 0; i < inputOrder.size(); ++i) {
        order.set(i, AxisCast(inputOrder[AxisCast(i, inputOrder.size())], inputOrder.size()));
    }
    return MakeConversion<arm_compute::NEPermute>(node.input(0), node.output(0), order);
}

template<typename D, typename I>
static void wrap_transpose(const D* data,
                           D* out,
                           const ngraph::Shape& data_shape,
                           const I* axes_order) {
    std::vector<std::int64_t> converted_axes_order(data_shape.size());
    if (axes_order == nullptr) {
        std::iota(converted_axes_order.begin(), converted_axes_order.end(), 0);
        std::reverse(converted_axes_order.begin(), converted_axes_order.end());
    } else {
        for (size_t i = 0; i < converted_axes_order.size(); ++i) {
            converted_axes_order[i] = static_cast<std::int64_t>(axes_order[i]);
        }
    }

    ngraph::Shape output_shape(converted_axes_order.size());
    std::transform(
        converted_axes_order.begin(),
        converted_axes_order.end(),
        output_shape.begin(),
        [&](const std::int64_t& v) {
            NGRAPH_CHECK(v >= 0,
                         "Negative values for transpose axes order are not supported.");
            NGRAPH_CHECK(v < int64_t(converted_axes_order.size()),
                         "Transpose axis ",
                         v,
                         " is out of shape range.");
            return data_shape[v];
        });

    ngraph::runtime::reference::transpose(reinterpret_cast<const char*>(data),
                                          reinterpret_cast<char*>(out),
                                          data_shape, sizeof(D),
                                          converted_axes_order.data(), output_shape);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Transpose& node) {
    auto make = [&] (auto refFunction) {
        if (ngraph::shape_size(node.get_input_shape(1)) == 0) {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.output(0),
                                        node.get_input_shape(0),
                                        nullptr);
        }
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.input(1));
    };
    return CallSwitch(
        AP_WRAP(make, wrap_transpose),
        node.input(0), allTypes,
        node.input(1), indexTypes);
}
} // namespace ArmPlugin
