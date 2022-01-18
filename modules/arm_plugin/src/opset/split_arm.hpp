// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmSplit : public Split {
public:
    OPENVINO_OP("ArmSplit", "arm_opset", Split);

    ArmSplit(const ngraph::Output<ngraph::Node>& data, const ngraph::Output<ngraph::Node>& axis, const size_t num_splits);

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
};
}  // namespace opset
}  // namespace ArmPlugin
