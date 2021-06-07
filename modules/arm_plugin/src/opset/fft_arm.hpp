// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmFFT : public ngraph::op::Op {
public:
    enum class Axis {
        axisX,
        axisY
    };

    static constexpr ngraph::NodeTypeInfo type_info{"ArmFFT", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info; }
    ArmFFT() = default;
    ~ArmFFT() override;

    ArmFFT(const ngraph::Output<ngraph::Node>& data, Axis axis, bool inverse);

    unsigned int get_arm_axis() const { return m_axis; }
    bool is_inverse() const { return m_inverse; }

    void validate_and_infer_types() override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
private:
    unsigned int m_axis;
    bool m_inverse;
};
}  // namespace opset
}  // namespace ArmPlugin
