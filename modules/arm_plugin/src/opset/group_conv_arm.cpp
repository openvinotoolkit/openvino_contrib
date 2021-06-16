// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_conv_arm.hpp"

using namespace ngraph;
using namespace ArmPlugin;

NGRAPH_RTTI_DEFINITION(opset::ArmGroupConvolution, "ArmGroupConvolution", 0);

opset::ArmGroupConvolution::~ArmGroupConvolution() {}

opset::ArmGroupConvolution::ArmGroupConvolution(const ngraph::Output<ngraph::Node>& data_batch,
                                                const ngraph::Output<ngraph::Node>& filters,
                                                const ngraph::Strides& strides,
                                                const ngraph::CoordinateDiff& pads_begin,
                                                const ngraph::CoordinateDiff& pads_end,
                                                const ngraph::Strides& dilations,
                                                const ngraph::op::PadType& auto_pad)
    : GroupConvolution{
        data_batch,
        filters,
        strides,
        pads_begin,
        pads_end,
        dilations,
        auto_pad} {
    constructor_validate_and_infer_types();
}

opset::ArmGroupConvolution::ArmGroupConvolution(const ngraph::Output<ngraph::Node>& data_batch,
                                                const ngraph::Output<ngraph::Node>& filters,
                                                const ngraph::Output<ngraph::Node>& bias,
                                                const ngraph::Strides& strides,
                                                const ngraph::CoordinateDiff& pads_begin,
                                                const ngraph::CoordinateDiff& pads_end,
                                                const ngraph::Strides& dilations,
                                                const ngraph::op::PadType& auto_pad)
    : GroupConvolution{
        data_batch,
        filters,
        strides,
        pads_begin,
        pads_end,
        dilations,
        auto_pad} {
    set_argument(2, bias);
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ArmPlugin::opset::ArmGroupConvolution::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    auto num_args = new_args.size();
    if (num_args == 2) {
        return std::make_shared<ArmGroupConvolution>(new_args.at(0),
                                                new_args.at(1),
                                                m_strides,
                                                m_pads_begin,
                                                m_pads_end,
                                                m_dilations,
                                                m_auto_pad);
    } else if (num_args == 3) {
        return std::make_shared<ArmGroupConvolution>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                m_strides,
                                                m_pads_begin,
                                                m_pads_end,
                                                m_dilations,
                                                m_auto_pad);
    } else {
        throw ngraph_error("Unsupported number of arguments for ArmGroupConvolution operation");
    }
}
