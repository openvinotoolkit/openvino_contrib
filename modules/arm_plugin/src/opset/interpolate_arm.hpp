// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmInterpolate : public Interpolate {
public:
    OPENVINO_OP("ArmInterpolate", "arm_opset", Interpolate);

    ArmInterpolate(const ngraph::Output<ngraph::Node>& image,
                   const ngraph::Output<ngraph::Node>& output_shape,
                   const ngraph::Output<ngraph::Node>& scales,
                   const Interpolate::InterpolateAttrs& attrs);

    ArmInterpolate(const ngraph::Output<ngraph::Node>& image,
                   const ngraph::Output<ngraph::Node>& output_shape,
                   const ngraph::Output<ngraph::Node>& scales,
                   const ngraph::Output<ngraph::Node>& axes,
                   const Interpolate::InterpolateAttrs& attrs);

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
private:
    Interpolate::InterpolateAttrs m_attrs;
};
}  // namespace opset
}  // namespace ArmPlugin
