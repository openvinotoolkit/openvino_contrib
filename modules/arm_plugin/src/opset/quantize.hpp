// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/variant.hpp>
#include "ngraph/coordinate_diff.hpp"
#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

struct ArmQuantize : public ngraph::op::Op {
    OPENVINO_OP("ArmQuantize", "arm_opset");
    ArmQuantize(const ngraph::Output<ngraph::Node>& data);
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    void validate_and_infer_types() override;
};

struct ArmDequantize : public ngraph::op::Op {
    OPENVINO_OP("ArmDequantize", "arm_opset");
    ArmDequantize(const ngraph::Output<ngraph::Node>& data);
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    void validate_and_infer_types() override;
};

}  // namespace opset
}  // namespace ArmPlugin
