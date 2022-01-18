// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "ngraph_opset.hpp"
#include "utils.hpp"
#include "quantize.hpp"

namespace ArmPlugin {
namespace opset {

class ArmGroupConvolution : public GroupConvolution {
public:
    OPENVINO_OP("ArmGroupConvolution", "arm_opset", GroupConvolution);

    ArmGroupConvolution(const ngraph::Output<ngraph::Node>& data_batch,
                        const ngraph::Output<ngraph::Node>& filters,
                        const ngraph::Strides& strides,
                        const ngraph::CoordinateDiff& pads_begin,
                        const ngraph::CoordinateDiff& pads_end,
                        const ngraph::Strides& dilations,
                        const ngraph::op::PadType& auto_pad);

    ArmGroupConvolution(const ngraph::Output<ngraph::Node>& data_batch,
                        const ngraph::Output<ngraph::Node>& filters,
                        const ngraph::Output<ngraph::Node>& bias,
                        const ngraph::Strides& strides,
                        const ngraph::CoordinateDiff& pads_begin,
                        const ngraph::CoordinateDiff& pads_end,
                        const ngraph::Strides& dilations,
                        const ngraph::op::PadType& auto_pad);

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
};
}  // namespace opset
}  // namespace ArmPlugin
