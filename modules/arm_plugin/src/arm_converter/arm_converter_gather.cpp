// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0



#include <arm_compute/runtime/NEON/functions/NEGather.h>
#include <ngraph/runtime/reference/gather.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template <> Converter::Conversion::Ptr Converter::Convert(const opset::Gather& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_output_shape(0),
                                    static_cast<size_t>(node.get_axis()),
                                    static_cast<size_t>(node.get_batch_dims()));
    };

    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::gather),
        node.input(0), allTypes,
        node.input(1), indexTypes);
}

template <> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::v1::Gather& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_output_shape(0),
                                    static_cast<size_t>(node.get_axis()),
                                    static_cast<size_t>(0));
    };

    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::gather),
        node.input(0), allTypes,
        node.input(1), indexTypes);
}

template <> Converter::Conversion::Ptr Converter::Convert(const opset::ArmGather& node) {
    auto axes = safe_cast<opset::Constant>(node.input_value(2).get_node_shared_ptr());

    if (node.get_input_shape(1).size() > 1) {
        IE_THROW() << "Supported Gather op with scalar or 1D indices only";
    }

    int axis = axes->cast_vector<int64_t>()[0];
    if (axis < 0) {
        axis += node.get_input_shape(0).size();
    }
    axis = AxisCast(axis, node.get_input_shape(0).size());
    return MakeConversion<arm_compute::NEGather>(node.input(0), node.input(1), node.output(0), axis);
}
}  //  namespace ArmPlugin
