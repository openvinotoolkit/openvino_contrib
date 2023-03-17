// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertConvBase: public ngraph::pass::MatcherPass {
protected:
    OPENVINO_OP("ConvertConvBase");
    template <class Conv, class ArmConv>
    ngraph::matcher_pass_callback convert_conv_to_arm_conv();
};

class ConvertSingleConvolutionToArm: public ConvertConvBase {
public:
    OPENVINO_OP("ConvertSingleConvolutionToArm");
    ConvertSingleConvolutionToArm();
};

class ConvertGroupConvolutionToArm: public ConvertConvBase {
public:
    OPENVINO_OP("ConvertGroupConvolutionToArm");
    ConvertGroupConvolutionToArm();
};

class ConvBiasFusionBase: public ngraph::pass::MatcherPass {
protected:
    OPENVINO_OP("ConvBiasFusionBase");
    template <class Conv, class Bias>
    void registerMatcher(const std::string& name);
};

class ConvAddFusion: public ConvBiasFusionBase {
public:
    OPENVINO_OP("ConvAddFusion");
    ConvAddFusion();
};

class GroupConvAddFusion: public ConvBiasFusionBase {
public:
    OPENVINO_OP("GroupConvAddFusion");
    GroupConvAddFusion();
};

class ConvBiasFusion: public ngraph::pass::GraphRewrite {
public:
    ConvBiasFusion() {
        add_matcher<ConvertSingleConvolutionToArm>();
        add_matcher<ConvertGroupConvolutionToArm>();

        add_matcher<ConvAddFusion>();
        add_matcher<GroupConvAddFusion>();
    }
};
}  // namespace pass
}  // namespace ArmPlugin
