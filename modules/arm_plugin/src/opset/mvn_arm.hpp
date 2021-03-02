// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmMVN : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"ArmMVN", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info; }
    ArmMVN() = default;
    ~ArmMVN() override;

    ArmMVN(const ngraph::Output<ngraph::Node>& data, float eps);

    float get_eps() const { return m_eps; }

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
private:
    float m_eps = 0.00001f;
};
}  // namespace opset
}  // namespace ArmPlugin
