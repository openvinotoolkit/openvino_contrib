// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv_bias_activ_fusion.hpp"

#include <memory>
#include <numeric>
#include <vector>

#include <ngraph/rt_info.hpp>

#include "opset/opset.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>

using namespace ArmPlugin;

enum Layout {N, C, H, W};
enum Inputs {Data, Weights, Bias};

template<class Conv, class Activation>
ngraph::matcher_pass_callback ArmPlugin::pass::ConvActivationFusionBase::fuse_conv_with_activation() {
    return [&](ngraph::pattern::Matcher& m) {
        auto activation = std::dynamic_pointer_cast<Activation>(m.get_match_root());
        if (!activation) {
            return false;
        }

        auto m_conv = std::dynamic_pointer_cast<Conv>(activation->input_value(0).get_node_shared_ptr());

        if (!m_conv) {
            return false;
        }

        if (m_conv->output(0).get_target_inputs().size() != 1) {
            return false;
        }

        opset::ActivationFunction func;
        float a = 0.0f, b = 0.0f;
        if (std::is_same<Activation, opset::Sigmoid>()) {
            func = opset::ActivationFunction::LOGISTIC;
        } else if (std::is_same<Activation, opset::Tanh>()) {
            func = opset::ActivationFunction::TANH;
        } else if (std::is_same<Activation, opset::Relu>()) {
            func = opset::ActivationFunction::RELU;
        } else if (std::is_same<Activation, opset::Abs>()) {
            func = opset::ActivationFunction::ABS;
        } else if (std::is_same<Activation, opset::Elu>()) {
            func = opset::ActivationFunction::ELU;
            auto elu = std::dynamic_pointer_cast<opset::Elu>(activation);
            if (!elu) {
                return false;
            }
            a = elu->get_alpha();
        } else if (std::is_same<Activation, opset::Sqrt>()) {
            func = opset::ActivationFunction::SQRT;
        } else if (std::is_same<Activation, opset::SoftPlus>()) {
            func = opset::ActivationFunction::SOFT_RELU;
        } else if (std::is_same<Activation, opset::HSwish>()) {
            func = opset::ActivationFunction::HARD_SWISH;
        } else if (std::is_same<Activation, opset::PRelu>()) {
            func = opset::ActivationFunction::LEAKY_RELU;
            auto prelu = std::dynamic_pointer_cast<opset::PRelu>(activation);
            if (!prelu) {
                return false;
            }
            a = dynamic_cast<const opset::Constant&>(*(prelu->input_value(1).get_node())).get_vector<float>()[0];
        } else if (std::is_same<Activation, opset::Clamp>()) {
            func = opset::ActivationFunction::LU_BOUNDED_RELU;
            auto clamp = std::dynamic_pointer_cast<opset::Clamp>(activation);
            if (!clamp) {
                return false;
            }
            a = clamp->get_max();
            b = clamp->get_min();
        } else {
            func = opset::ActivationFunction::IDENTITY;
        }

        std::shared_ptr<ngraph::Node> conv_activ;
        if (m_conv->get_input_size() == 2) {
            conv_activ = std::make_shared<Conv>(
                    m_conv->input_value(Inputs::Data),
                    m_conv->input_value(Inputs::Weights),
                    m_conv->get_strides(),
                    m_conv->get_pads_begin(),
                    m_conv->get_pads_end(),
                    m_conv->get_dilations(),
                    m_conv->get_auto_pad(),
                    opset::ActivationInfo{func, a, b});
        } else {
            conv_activ = std::make_shared<Conv>(
                    m_conv->input_value(Inputs::Data),
                    m_conv->input_value(Inputs::Weights),
                    m_conv->input_value(Inputs::Bias),
                    m_conv->get_strides(),
                    m_conv->get_pads_begin(),
                    m_conv->get_pads_end(),
                    m_conv->get_dilations(),
                    m_conv->get_auto_pad(),
                    opset::ActivationInfo{func, a, b});
        }

        ngraph::copy_runtime_info({m_conv, activation}, conv_activ);
        conv_activ->set_friendly_name(activation->get_friendly_name());
        ngraph::replace_node(activation, conv_activ);
        return true;
    };
}

template <class Conv>
ngraph::matcher_pass_callback ArmPlugin::pass::ConvBiasFusionBase::fuse_conv_with_bias() {
   return [&](ngraph::pattern::Matcher& m) {
        auto eltwise = std::dynamic_pointer_cast<opset::Add>(m.get_match_root());
        if (!eltwise) {
            return false;
        }

        int conv_idx = 0;
        auto m_conv = std::dynamic_pointer_cast<Conv>(eltwise->input_value(conv_idx).get_node_shared_ptr());
        if (!m_conv) {
            conv_idx = 1;
            m_conv = std::dynamic_pointer_cast<Conv>(eltwise->input_value(conv_idx).get_node_shared_ptr());
        }

        if (!m_conv) {
            return false;
        }

        auto bias = eltwise->input_value(1 - conv_idx);
        // TODO: check that constant can be scalar and do not match [1, C, 1, 1] layout
        const auto bias_shape = bias.get_shape();
        const auto output_pshape = m_conv->get_output_partial_shape(0);

        if (output_pshape.rank().is_dynamic() || output_pshape[Layout::C].is_dynamic()) {
            return false;
        }

        const auto channel_dim = output_pshape[Layout::C].get_length();
        int broad_dims = m_conv->get_shape().size() - bias_shape.size();
        int channel_ind = Layout::C - broad_dims;
        if (channel_ind < 0 || bias_shape[channel_ind] != channel_dim || ngraph::shape_size(bias_shape) != channel_dim) {
            return false;
        }

        ngraph::Output<ngraph::Node> new_bias(bias);

        if (bias_shape.size() > 1) {
            new_bias = std::make_shared<opset::Reshape>(bias,
                    opset::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {channel_dim}), true);
        }

        if (m_conv->output(0).get_target_inputs().size() != 1) {
            return false;
        }

        if (m_conv->inputs().size() == 3) {
            new_bias = std::make_shared<opset::Add>(new_bias, m_conv->input_value(Inputs::Bias));
        }

        auto new_conv = std::make_shared<Conv>(
                    m_conv->input_value(Inputs::Data),
                    m_conv->input_value(Inputs::Weights),
                    new_bias,
                    m_conv->get_strides(),
                    m_conv->get_pads_begin(),
                    m_conv->get_pads_end(),
                    m_conv->get_dilations(),
                    m_conv->get_auto_pad(),
                    opset::ActivationInfo{});

        ngraph::copy_runtime_info({m_conv, eltwise}, new_conv);
        new_conv->set_friendly_name(eltwise->get_friendly_name());
        ngraph::replace_node(eltwise, new_conv);
        return true;
    };
}

template <class Conv, class ArmConv>
ngraph::matcher_pass_callback ArmPlugin::pass::ConvertConvBase::convert_conv_to_arm_conv() {
    return [&](ngraph::pattern::Matcher& m) {
        auto m_conv = std::dynamic_pointer_cast<Conv>(m.get_match_root());
        if (!m_conv) {
            return false;
        }
        if (m_conv->output(0).get_target_inputs().size() != 1) {
            return false;
        }

        auto conv_arm = std::make_shared<ArmConv>(
                    m_conv->input_value(Inputs::Data),
                    m_conv->input_value(Inputs::Weights),
                    m_conv->get_strides(),
                    m_conv->get_pads_begin(),
                    m_conv->get_pads_end(),
                    m_conv->get_dilations(),
                    m_conv->get_auto_pad(),
                    opset::ActivationInfo{});

        ngraph::copy_runtime_info(m_conv, conv_arm);
        conv_arm->set_friendly_name(m_conv->get_friendly_name());
        ngraph::replace_node(m_conv, conv_arm);
        return true;
    };
}

// ----------------------------------------ConvertConvolution----------------------------------------

ArmPlugin::pass::ConvertSingleConvolutionToArm::ConvertSingleConvolutionToArm() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<opset::Convolution>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                        ngraph::pattern::has_static_shape()), "ConvertConvolutionToArm");
    register_matcher(m, convert_conv_to_arm_conv<opset::Convolution, opset::ArmConvolution>());
}

ArmPlugin::pass::ConvertGroupConvolutionToArm::ConvertGroupConvolutionToArm() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::GroupConvolution>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                                 ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                                 ngraph::pattern::has_static_shape()), "ConvertGroupConvolutionToArm");
    register_matcher(m, convert_conv_to_arm_conv<opset::GroupConvolution, opset::ArmGroupConvolution>());
}

// ------------------------------------------ConvBiasFusion------------------------------------------

ArmPlugin::pass::ConvBiasFusion::ConvBiasFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::ArmConvolution>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                               ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                               ngraph::pattern::has_static_shape()), "ConvBiasFusion");
    register_matcher(m, fuse_conv_with_bias<opset::ArmConvolution>());
}

ArmPlugin::pass::GroupConvBiasFusion::GroupConvBiasFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::ArmGroupConvolution>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                                        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                                        ngraph::pattern::has_static_shape()), "GroupConvBiasFusion");
    register_matcher(m, fuse_conv_with_bias<opset::ArmGroupConvolution>());
}

// ---------------------------------------ConvActivationFusion---------------------------------------

ArmPlugin::pass::ConvSigmoidFusion::ConvSigmoidFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::Sigmoid>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                            ngraph::pattern::has_static_shape()), "ConvSigmoidFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmConvolution, opset::Sigmoid>());
}

ArmPlugin::pass::ConvTanhFusion::ConvTanhFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::Tanh>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                         ngraph::pattern::has_static_shape()), "ConvTanhFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmConvolution, opset::Tanh>());
}

ArmPlugin::pass::ConvReluFusion::ConvReluFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::Relu>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                         ngraph::pattern::has_static_shape()), "ConvReluFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmConvolution, opset::Relu>());
}

ArmPlugin::pass::ConvClampFusion::ConvClampFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::Clamp>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                          ngraph::pattern::has_static_shape()), "ConvClampFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmConvolution, opset::Clamp>());
}

ArmPlugin::pass::ConvAbsFusion::ConvAbsFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::Abs>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                        ngraph::pattern::has_static_shape()), "ConvAbsFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmConvolution, opset::Abs>());
}

ArmPlugin::pass::ConvEluFusion::ConvEluFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::Elu>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                        ngraph::pattern::has_static_shape()), "ConvEluFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmConvolution, opset::Elu>());
}

ArmPlugin::pass::ConvSqrtFusion::ConvSqrtFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::Sqrt>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                         ngraph::pattern::has_static_shape()), "ConvSqrtFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmConvolution, opset::Sqrt>());
}

ArmPlugin::pass::ConvSoftPlusFusion::ConvSoftPlusFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::SoftPlus>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                             ngraph::pattern::has_static_shape()), "ConvSoftPlusFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmConvolution, opset::SoftPlus>());
}

ArmPlugin::pass::ConvHSwishFusion::ConvHSwishFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::HSwish>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                           ngraph::pattern::has_static_shape()), "ConvHSwishFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmConvolution, opset::HSwish>());
}

ArmPlugin::pass::ConvPReluFusion::ConvPReluFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::PRelu>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                          ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                          ngraph::pattern::has_static_shape()), "ConvPReluFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmConvolution, opset::PRelu>());
}

ArmPlugin::pass::GroupConvSigmoidFusion::GroupConvSigmoidFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::Sigmoid>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                            ngraph::pattern::has_static_shape()), "GroupConvSigmoidFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmGroupConvolution, opset::Sigmoid>());
}

ArmPlugin::pass::GroupConvTanhFusion::GroupConvTanhFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::Tanh>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                         ngraph::pattern::has_static_shape()), "GroupConvTanhFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmGroupConvolution, opset::Tanh>());
}

ArmPlugin::pass::GroupConvReluFusion::GroupConvReluFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::Relu>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                         ngraph::pattern::has_static_shape()), "GroupConvReluFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmGroupConvolution, opset::Relu>());
}

ArmPlugin::pass::GroupConvClampFusion::GroupConvClampFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::Clamp>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                          ngraph::pattern::has_static_shape()), "GroupConvClampFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmGroupConvolution, opset::Clamp>());
}

ArmPlugin::pass::GroupConvAbsFusion::GroupConvAbsFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::Abs>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                        ngraph::pattern::has_static_shape()), "GroupConvAbsFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmGroupConvolution, opset::Abs>());
}

ArmPlugin::pass::GroupConvEluFusion::GroupConvEluFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::Elu>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                        ngraph::pattern::has_static_shape()), "GroupConvEluFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmGroupConvolution, opset::Elu>());
}

ArmPlugin::pass::GroupConvSqrtFusion::GroupConvSqrtFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::Sqrt>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                         ngraph::pattern::has_static_shape()), "GroupConvSqrtFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmGroupConvolution, opset::Sqrt>());
}

ArmPlugin::pass::GroupConvSoftPlusFusion::GroupConvSoftPlusFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::SoftPlus>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                             ngraph::pattern::has_static_shape()), "GroupConvSoftPlusFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmGroupConvolution, opset::SoftPlus>());
}

ArmPlugin::pass::GroupConvHSwishFusion::GroupConvHSwishFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::HSwish>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                           ngraph::pattern::has_static_shape()), "GroupConvHSwishFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmGroupConvolution, opset::HSwish>());
}

ArmPlugin::pass::GroupConvPReluFusion::GroupConvPReluFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::PRelu>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                          ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                          ngraph::pattern::has_static_shape()), "GroupConvPReluFusion");
    register_matcher(m, fuse_conv_with_activation<opset::ArmGroupConvolution, opset::PRelu>());
}
