// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concat_arm.hpp"

using namespace ngraph;
using namespace ArmPlugin;

opset::ArmConcat::ArmConcat(const ngraph::OutputVector& args, int64_t axis)
    : Concat{args, axis} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ArmPlugin::opset::ArmConcat::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    return std::make_shared<ArmConcat>(new_args, m_axis);
}
