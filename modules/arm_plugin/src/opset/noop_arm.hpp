// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ArmPlugin {
namespace opset {
struct ArmNoOp : public ngraph::op::Op {
    NGRAPH_RTTI_DECLARATION;
    ArmNoOp() = default;
    ArmNoOp(const ngraph::Output<ngraph::Node>& arg);
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
};
} // namespace opset
} // namespace ArmPlugin
