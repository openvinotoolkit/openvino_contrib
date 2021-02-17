// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmGroupConvolution : public GroupConvolution {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"ArmGroupConvolution", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info; }
    ArmGroupConvolution() = default;
    ~ArmGroupConvolution() override;

    ArmGroupConvolution(const ngraph::Output<ngraph::Node>& data_batch,
                        const ngraph::Output<ngraph::Node>& filters,
                        const ngraph::Strides& strides,
                        const ngraph::CoordinateDiff& pads_begin,
                        const ngraph::CoordinateDiff& pads_end,
                        const ngraph::Strides& dilations,
                        const ngraph::op::PadType& auto_pad,
                        const ActivationInfo& activation);

    ArmGroupConvolution(const ngraph::Output<ngraph::Node>& data_batch,
                        const ngraph::Output<ngraph::Node>& filters,
                        const ngraph::Output<ngraph::Node>& bias,
                        const ngraph::Strides& strides,
                        const ngraph::CoordinateDiff& pads_begin,
                        const ngraph::CoordinateDiff& pads_end,
                        const ngraph::Strides& dilations,
                        const ngraph::op::PadType& auto_pad,
                        const ActivationInfo& activation);

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    const ActivationInfo& get_info() const { return m_info; }

private:
    ActivationInfo m_info;
};
}  // namespace opset
}  // namespace ArmPlugin
