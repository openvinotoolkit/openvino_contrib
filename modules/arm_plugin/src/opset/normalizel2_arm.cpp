// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalizel2_arm.hpp"

using namespace ngraph;
using namespace ArmPlugin;

opset::ArmNormalizeL2::ArmNormalizeL2(const Output<Node>& data,
                                      const Output<Node>& axes,
                                      float eps,
                                      op::EpsMode eps_mode)
    : NormalizeL2{data, axes, eps, eps_mode} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ArmPlugin::opset::ArmNormalizeL2::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    auto num_args = new_args.size();
    if (num_args == 2) {
        return std::make_shared<ArmNormalizeL2>(new_args.at(0), new_args.at(1), m_eps, m_eps_mode);
    } else {
        throw ngraph_error("Unsupported number of arguments for ArmMVN operation");
    }
}
