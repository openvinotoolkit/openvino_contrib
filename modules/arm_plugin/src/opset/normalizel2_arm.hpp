// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmNormalizeL2 : public NormalizeL2 {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"ArmNormalizeL2", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info; }
    ArmNormalizeL2() = default;
    ~ArmNormalizeL2() override;

    ArmNormalizeL2(const ngraph::Output<ngraph::Node>& data,
                   const ngraph::Output<ngraph::Node>& axes,
                   float eps,
                   ngraph::op::EpsMode eps_mode);

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
};
}  // namespace opset
}  // namespace ArmPlugin
