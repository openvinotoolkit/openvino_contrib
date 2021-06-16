// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_arm.hpp"

using namespace ngraph;
using namespace ArmPlugin;

NGRAPH_RTTI_DEFINITION(opset::ArmMVN, "ArmMVN", 0);

opset::ArmMVN::~ArmMVN() {}

opset::ArmMVN::ArmMVN(const ngraph::Output<ngraph::Node>& data, float eps)
    : Op({data}), m_eps(eps) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> opset::ArmMVN::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    auto num_args = new_args.size();
    if (num_args == 1) {
        return std::make_shared<ArmMVN>(new_args.at(0), m_eps);
    } else {
        throw ngraph_error("Unsupported number of arguments for ArmMVN operation");
    }
}
