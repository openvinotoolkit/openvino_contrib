// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose_arm.hpp"

using namespace ngraph;
using namespace ArmPlugin;

opset::ArmTranspose::ArmTranspose(const ngraph::Output<ngraph::Node>& arg, const ngraph::Output<ngraph::Node>& input_order)
    : Transpose{arg, input_order} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ArmPlugin::opset::ArmTranspose::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    auto num_args = new_args.size();
    if (num_args == 2) {
        return std::make_shared<ArmTranspose>(new_args.at(0), new_args.at(1));
    } else {
        throw ngraph_error("Unsupported number of arguments for ArmTranspose operation");
    }
}
