// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_arm.hpp"

using namespace ngraph;
using namespace ArmPlugin;

opset::ArmConvert::ArmConvert(const ngraph::Output<ngraph::Node>& data,
                              const ngraph::element::Type& destination_type)
    : Convert{data, destination_type} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ArmPlugin::opset::ArmConvert::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    auto num_args = new_args.size();
    if (num_args == 1) {
        return std::make_shared<ArmConvert>(new_args.at(0),
                                            m_destination_type);
    } else {
        throw ngraph_error("Unsupported number of arguments for ArmConvert operation");
    }
}
