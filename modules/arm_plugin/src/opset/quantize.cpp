// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <details/ie_exception.hpp>
#include "quantize.hpp"
#include <half/half.hpp>

using namespace ArmPlugin;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(opset::ArmQuantize, "ArmQuantize", 0);

opset::ArmQuantize::ArmQuantize(const ngraph::Output<ngraph::Node>& data,
                                const ngraph::Output<ngraph::Node>& input_low,
                                const ngraph::Output<ngraph::Node>& input_high,
                                const ngraph::Output<ngraph::Node>& output_low,
                                const ngraph::Output<ngraph::Node>& output_high,
                                std::size_t levels,
                                const ngraph::op::AutoBroadcastSpec& auto_broadcast) :
    FakeQuantize{data, input_low, input_high, output_low, output_high, levels, auto_broadcast} {}

opset::ArmQuantize::~ArmQuantize() {}

std::shared_ptr<Node> opset::ArmQuantize::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<ArmQuantize>(new_args.at(0), // X
                                         new_args.at(1), // input_low
                                         new_args.at(2), // input_high
                                         new_args.at(3), // output_low
                                         new_args.at(4), // output_high
                                         get_levels(),
                                         get_auto_broadcast());
}

NGRAPH_RTTI_DEFINITION(opset::ArmDequantize, "ArmDequantize", 0);

opset::ArmDequantize::ArmDequantize(const ngraph::Output<ngraph::Node>& data,
                                    const ngraph::Output<ngraph::Node>& input_low,
                                    const ngraph::Output<ngraph::Node>& input_high,
                                    const ngraph::Output<ngraph::Node>& output_low,
                                    const ngraph::Output<ngraph::Node>& output_high,
                                    std::size_t levels,
                                    const ngraph::op::AutoBroadcastSpec& auto_broadcast) :
    FakeQuantize{data, input_low, input_high, output_low, output_high, levels, auto_broadcast} {}

opset::ArmDequantize::~ArmDequantize() {}

std::shared_ptr<Node> opset::ArmDequantize::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<ArmDequantize>(new_args.at(0), // X
                                           new_args.at(1), // input_low
                                           new_args.at(2), // input_high
                                           new_args.at(3), // output_low
                                           new_args.at(4), // output_high
                                           get_levels(),
                                           get_auto_broadcast());
}
