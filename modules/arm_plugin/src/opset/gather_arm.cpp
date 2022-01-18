// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_arm.hpp"

using namespace ngraph;
using namespace ArmPlugin;

opset::ArmGather::ArmGather(const ngraph::Output<ngraph::Node>& data,
                            const ngraph::Output<ngraph::Node>& indices,
                            const ngraph::Output<ngraph::Node>& axes)
    : Gather({data, indices, axes}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> opset::ArmGather::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    auto num_args = new_args.size();
    if (num_args == 3) {
        return std::make_shared<ArmGather>(new_args.at(0), new_args.at(1), new_args.at(2));
    } else {
        throw ngraph_error("Unsupported number of arguments for ArmGather operation");
    }
}
