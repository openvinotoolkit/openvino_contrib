// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmTranspose : public Transpose {
public:
    OPENVINO_OP("ArmTranspose", "arm_opset", Transpose);
    ArmTranspose(const ngraph::Output<ngraph::Node>& arg, const ngraph::Output<ngraph::Node>& input_order);
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
};
}  // namespace opset
}  // namespace ArmPlugin
