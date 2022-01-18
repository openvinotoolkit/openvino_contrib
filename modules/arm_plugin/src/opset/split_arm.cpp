// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "split_arm.hpp"

using namespace ngraph;
using namespace ArmPlugin;

opset::ArmSplit::ArmSplit(const ngraph::Output<ngraph::Node>& data, const ngraph::Output<ngraph::Node>& axis, const size_t num_splits)
    : Split{data, axis, num_splits} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ArmPlugin::opset::ArmSplit::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    return std::make_shared<ArmSplit>(new_args.at(0), new_args.at(1), get_num_splits());
}
