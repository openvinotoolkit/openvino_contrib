// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <details/ie_exception.hpp>
#include "quantize.hpp"
#include <half/half.hpp>

using namespace ArmPlugin;
using namespace ngraph;

opset::ArmQuantize::ArmQuantize(const ngraph::Output<ngraph::Node>& data) : Op{{data}} {}

std::shared_ptr<Node> opset::ArmQuantize::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<ArmQuantize>(new_args.at(0));
}

void opset::ArmQuantize::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

opset::ArmDequantize::ArmDequantize(const ngraph::Output<ngraph::Node>& data) : Op{{data}} {}

std::shared_ptr<Node> opset::ArmDequantize::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<ArmDequantize>(new_args.at(0));
}

void opset::ArmDequantize::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}
