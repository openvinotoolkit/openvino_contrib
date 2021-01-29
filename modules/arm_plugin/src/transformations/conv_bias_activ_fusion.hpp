// Copyright (C) 2018-2021 Intel Corporation
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

class ConvActivationFusionBase: public ngraph::pass::MatcherPass {
protected:
    template <class Conv, class Activation>
    ngraph::matcher_pass_callback fuse_conv_with_activation();
};

class ConvSigmoidFusion: public ConvActivationFusionBase {
public:
    ConvSigmoidFusion();
};

class ConvTanhFusion: public ConvActivationFusionBase {
public:
    ConvTanhFusion();
};

class ConvReluFusion: public ConvActivationFusionBase {
public:
    ConvReluFusion();
};

class ConvAbsFusion: public ConvActivationFusionBase {
public:
    ConvAbsFusion();
};

class ConvSqrtFusion: public ConvActivationFusionBase {
public:
    ConvSqrtFusion();
};

class ConvPReluFusion: public ConvActivationFusionBase {
public:
    ConvPReluFusion();
};

class ConvHSwishFusion: public ConvActivationFusionBase {
public:
    ConvHSwishFusion();
};

class ConvEluFusion: public ConvActivationFusionBase {
public:
    ConvEluFusion();
};

class ConvClampFusion: public ConvActivationFusionBase {
public:
    ConvClampFusion();
};

class ConvSoftPlusFusion: public ConvActivationFusionBase {
public:
    ConvSoftPlusFusion();
};


class GroupConvSigmoidFusion: public ConvActivationFusionBase {
public:
    GroupConvSigmoidFusion();
};

class GroupConvTanhFusion: public ConvActivationFusionBase {
public:
    GroupConvTanhFusion();
};

class GroupConvReluFusion: public ConvActivationFusionBase {
public:
    GroupConvReluFusion();
};

class GroupConvAbsFusion: public ConvActivationFusionBase {
public:
    GroupConvAbsFusion();
};

class GroupConvSqrtFusion: public ConvActivationFusionBase {
public:
    GroupConvSqrtFusion();
};

class GroupConvPReluFusion: public ConvActivationFusionBase {
public:
    GroupConvPReluFusion();
};

class GroupConvHSwishFusion: public ConvActivationFusionBase {
public:
    GroupConvHSwishFusion();
};

class GroupConvEluFusion: public ConvActivationFusionBase {
public:
    GroupConvEluFusion();
};

class GroupConvClampFusion: public ConvActivationFusionBase {
public:
    GroupConvClampFusion();
};

class GroupConvSoftPlusFusion: public ConvActivationFusionBase {
public:
    GroupConvSoftPlusFusion();
};

class ConvBiasActivationFusion: public ngraph::pass::GraphRewrite {
public:
    ConvBiasActivationFusion() {
        add_matcher<ConvertSingleConvolutionToArm>();
        add_matcher<ConvertGroupConvolutionToArm>();

        add_matcher<ConvBiasFusion>();
        add_matcher<GroupConvBiasFusion>();

        add_matcher<ConvSigmoidFusion>();
        add_matcher<ConvTanhFusion>();
        add_matcher<ConvReluFusion>();
        add_matcher<ConvAbsFusion>();
        add_matcher<ConvPReluFusion>();
        add_matcher<ConvSqrtFusion>();
        add_matcher<ConvHSwishFusion>();
        add_matcher<ConvSoftPlusFusion>();
        add_matcher<ConvClampFusion>();
        add_matcher<ConvEluFusion>();

        add_matcher<GroupConvSigmoidFusion>();
        add_matcher<GroupConvTanhFusion>();
        add_matcher<GroupConvReluFusion>();
        add_matcher<GroupConvAbsFusion>();
        add_matcher<GroupConvPReluFusion>();
        add_matcher<GroupConvSqrtFusion>();
        add_matcher<GroupConvHSwishFusion>();
        add_matcher<GroupConvSoftPlusFusion>();
        add_matcher<GroupConvClampFusion>();
        add_matcher<GroupConvEluFusion>();
    }
};
}  // namespace pass
}  // namespace ArmPlugin
