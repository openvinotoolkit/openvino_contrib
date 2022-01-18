// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmNormalizeL2 : public NormalizeL2 {
public:
    OPENVINO_OP("ArmNormalizeL2", "arm_opset", NormalizeL2);
    ArmNormalizeL2(const ngraph::Output<ngraph::Node>& data,
                   const ngraph::Output<ngraph::Node>& axes,
                   float eps,
                   ngraph::op::EpsMode eps_mode);
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
};
}  // namespace opset
}  // namespace ArmPlugin
