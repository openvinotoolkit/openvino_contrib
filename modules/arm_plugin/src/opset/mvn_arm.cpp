// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_arm.hpp"

using namespace ngraph;
using namespace ArmPlugin;

opset::ArmMVN::ArmMVN(const ngraph::Output<ngraph::Node>& data, float eps)
    : Op({data}), m_eps(eps) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> opset::ArmMVN::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    auto num_args = new_args.size();
    OPENVINO_ASSERT(num_args == 1, "Unsupported number of arguments for ArmMVN operation: ", num_args);
    return std::make_shared<ArmMVN>(new_args.at(0), m_eps);
}
