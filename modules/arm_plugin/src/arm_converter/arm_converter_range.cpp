// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <arm_compute/runtime/NEON/functions/NERange.h>
#include <openvino/core/validation_util.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Range& node) {
    std::cout << "Range converter - start" << std::endl;
    enum Input {Start, Stop, Step};
    if (ov::get_constant_from_source(node.input_value(Input::Start)) == nullptr) {
        std::cout << get_constant_from_source(node.input_value(0)) << std::endl;
        std::cout << node.input_value(1) << std::endl;
        std::cout << node.input_value(2) << std::endl;
    }
    auto start = ov::get_constant_from_source(node.input_value(Input::Start))->cast_vector<float>()[0];
    std::cout << "Range converter - finish" << std::endl;
    auto stop = ov::get_constant_from_source(node.input_value(Input::Stop))->cast_vector<float>()[0];
    auto step = ov::get_constant_from_source(node.input_value(Input::Step))->cast_vector<float>()[0];
    return MakeConversion<arm_compute::NERange>(node.output(0), start, stop, step);
}
} // namespace ArmPlugin