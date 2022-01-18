// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <arm_compute/runtime/NEON/functions/NESplit.h>
#include <arm_compute/runtime/NEON/functions/NECopy.h>
#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/split.hpp>

namespace ArmPlugin {
template <typename T>
static void wrap_split(T* in,
                       std::vector<Argument<Tensor*>> outputs,
                       const ngraph::Shape& inp_shape,
                       std::size_t elem_size,
                       std::int64_t axis,
                       std::size_t num_splits) {
    std::vector<char*> char_outputs;
    for (auto& output : outputs) {
        char_outputs.push_back(const_cast<char*>(reinterpret_cast<const char*>(static_cast<const T*>(output))));
    }
    ngraph::runtime::reference::split(reinterpret_cast<char*>(in),
                                      inp_shape,
                                      elem_size,
                                      axis,
                                      num_splits,
                                      char_outputs.data());
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Split& node) {
    auto make = [&] (auto refFunction) {
        auto axis = static_cast<std::int64_t>(safe_cast<opset::Constant>(node.input_value(1).get_node())->cast_vector<std::int32_t>()[0]);
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.outputs(),
                                    node.input(0).get_shape(),
                                    node.get_input_element_type(0).size(),
                                    axis,
                                    node.get_num_splits());
    };

    return CallSwitch(
        AP_WRAP(make, wrap_split),
        node.input(0), allTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmSplit& node) {
    size_t numDimensions = node.get_output_shape(0).size();
    int axis = safe_cast<opset::Constant>(
        node.input_value(1).get_node())->cast_vector<std::int32_t>()[0];
    if (axis < 0) {
        axis += numDimensions;
    }
    if (node.get_output_size() == 1) {
        return MakeConversion<arm_compute::NECopy>(node.input(0), node.output(0));
    }
    return MakeConversion<arm_compute::NESplit>(node.input(0), node.outputs(),
        static_cast<unsigned int>(AxisCast(axis, numDimensions)));
}
}  //  namespace ArmPlugin