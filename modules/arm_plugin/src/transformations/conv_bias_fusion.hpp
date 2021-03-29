// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {
class ConvertConvBase: public ngraph::pass::MatcherPass {
protected:
    template <class Conv, class ArmConv>
    ngraph::matcher_pass_callback convert_conv_to_arm_conv();
};

class ConvertSingleConvolutionToArm: public ConvertConvBase {
public:
    ConvertSingleConvolutionToArm();
};

class ConvertGroupConvolutionToArm: public ConvertConvBase {
public:
    ConvertGroupConvolutionToArm();
};

class ConvBiasFusionBase: public ngraph::pass::MatcherPass {
protected:
    template <class Conv>
    ngraph::matcher_pass_callback fuse_conv_with_bias();
};

class ConvBiasFusion: public ConvBiasFusionBase {
public:
    ConvBiasFusion();
};

class GroupConvBiasFusion: public ConvBiasFusionBase {
public:
    GroupConvBiasFusion();
};

class ConvBiasActivationFusion: public ngraph::pass::GraphRewrite {
public:
    ConvBiasActivationFusion() {
        add_matcher<ConvertSingleConvolutionToArm>();
        add_matcher<ConvertGroupConvolutionToArm>();

        add_matcher<ConvBiasFusion>();
        add_matcher<GroupConvBiasFusion>();
    }
};
}  // namespace pass
}  // namespace ArmPlugin
