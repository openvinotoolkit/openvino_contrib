// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmConvert : public Convert {
public:
    OPENVINO_OP("ArmConvert", "arm_opset", Convert);
    ArmConvert(const ngraph::Output<ngraph::Node>& data, const ngraph::element::Type& destination_type);
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
};
}  // namespace opset
}  // namespace ArmPlugin
