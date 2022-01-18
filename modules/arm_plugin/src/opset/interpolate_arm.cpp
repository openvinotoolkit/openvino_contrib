// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate_arm.hpp"

using namespace ngraph;
using namespace ArmPlugin;

opset::ArmInterpolate::ArmInterpolate(const ngraph::Output<ngraph::Node>& image,
                                      const ngraph::Output<ngraph::Node>& output_shape,
                                      const ngraph::Output<ngraph::Node>& scales,
                                      const Interpolate::InterpolateAttrs& attrs)
    : Interpolate{
        image,
        output_shape,
        scales,
        attrs}, m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

opset::ArmInterpolate::ArmInterpolate(const ngraph::Output<ngraph::Node>& image,
                                      const ngraph::Output<ngraph::Node>& output_shape,
                                      const ngraph::Output<ngraph::Node>& scales,
                                      const ngraph::Output<ngraph::Node>& axes,
                                      const Interpolate::InterpolateAttrs& attrs)
    : Interpolate{
        image,
        output_shape,
        scales,
        axes,
        attrs}, m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ArmPlugin::opset::ArmInterpolate::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    auto num_args = new_args.size();
    if (num_args == 3) {
        return std::make_shared<ArmInterpolate>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                m_attrs);
    } else if (num_args == 4) {
        return std::make_shared<ArmInterpolate>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                new_args.at(3),
                                                m_attrs);
    } else {
        throw ngraph_error("Unsupported number of arguments for ArmInterpolate operation");
    }
}
