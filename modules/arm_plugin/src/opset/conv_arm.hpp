// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "ngraph_opset.hpp"
#include "utils.hpp"
#include "quantize.hpp"

namespace ArmPlugin {
namespace opset {

class ArmConvolution : public Convolution {
public:
    OPENVINO_OP("ArmConvolution", "arm_opset", Convolution);
    ArmConvolution(const ngraph::Output<ngraph::Node>& data_batch,
                   const ngraph::Output<ngraph::Node>& filters,
                   const ngraph::Strides& strides,
                   const ngraph::CoordinateDiff& pads_begin,
                   const ngraph::CoordinateDiff& pads_end,
                   const ngraph::Strides& dilations,
                   const ngraph::op::PadType& auto_pad,
                   const ngraph::PartialShape& output_shape = {});

    ArmConvolution(const ngraph::Output<ngraph::Node>& data_batch,
                   const ngraph::Output<ngraph::Node>& filters,
                   const ngraph::Output<ngraph::Node>& bias,
                   const ngraph::Strides& strides,
                   const ngraph::CoordinateDiff& pads_begin,
                   const ngraph::CoordinateDiff& pads_end,
                   const ngraph::Strides& dilations,
                   const ngraph::op::PadType& auto_pad,
                   const ngraph::PartialShape& output_shape = {});

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    void validate_and_infer_types() override;

private:
    ngraph::PartialShape m_output_shape;
};

}  // namespace opset
}  // namespace ArmPlugin
