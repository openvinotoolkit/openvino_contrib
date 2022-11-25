// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "slice_arm.hpp"
#include "ngraph/validation_util.hpp"

using namespace ngraph;
using namespace ArmPlugin;

opset::ArmSlice::~ArmSlice() {}

opset::ArmSlice::ArmSlice(const ngraph::Output<ngraph::Node>& data,
                                        const ngraph::Output<ngraph::Node>& begin,
                                        const ngraph::Output<ngraph::Node>& end,
                                        const ngraph::Output<ngraph::Node>& step,
                                        const ngraph::Output<ngraph::Node>& axes)
    : ngraph::op::v8::Slice{data, begin, end, step, axes } {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ArmPlugin::opset::ArmSlice::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    auto num_args = new_args.size();
    if (num_args == 4) {
        return std::make_shared<ArmSlice>(new_args.at(0),
                                          new_args.at(1),
                                          new_args.at(2),
                                          new_args.at(3),
                                          new_args.at(4));
    } else {
        throw ngraph_error("Unsupported number of arguments for ArmSlice operation");
    }
}
