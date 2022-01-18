// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmGather : public Gather {
public:
    OPENVINO_OP("ArmGather", "arm_opset", Gather);
    ArmGather(const ngraph::Output<ngraph::Node>& data,
              const ngraph::Output<ngraph::Node>& indices,
              const ngraph::Output<ngraph::Node>& axes);
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
};
}  // namespace opset
}  // namespace ArmPlugin
