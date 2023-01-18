// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include <ngraph/opsets/opset7.hpp>
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmSlice : public ngraph::op::v8::Slice {
public:
    OPENVINO_OP("ArmSlice", "arm_opset", ngraph::op::v8::Slice);

    ~ArmSlice() override;

    ArmSlice(const ngraph::Output<ngraph::Node>& data,
                    const ngraph::Output<ngraph::Node>& begin,
                    const ngraph::Output<ngraph::Node>& end,
                    const ngraph::Output<ngraph::Node>& step,
                    const ngraph::Output<ngraph::Node>& axes);

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
};
}  // namespace opset
}  // namespace ArmPlugin
