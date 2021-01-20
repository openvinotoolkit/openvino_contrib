// Copyright (C) 2018-2021 Intel Corporation
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

template<typename Conv, typename Activation>
static auto addFuseActivationMatcher(pass::ConvBiasActivationFusion* pass) {
    auto conv = ngraph::pattern::wrap_type<Conv>();
    auto activation = ngraph::pattern::wrap_type<Activation>({conv});
    auto m = std::make_shared<ngraph::pattern::Matcher>(activation, (std::string {Conv::type_info.name} + "ActivationFusion"));

    pass->add_matcher(m, [] (ngraph::pattern::Matcher& m) {
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
            a = dynamic_cast<const opset::Constant&>(*(prelu->input_value(1).get_node())).get_vector<float>()[0];
        } else if (std::is_same<Activation, opset::Clamp>()) {
            func = opset::ActivationFunction::LU_BOUNDED_RELU;
            auto clamp = std::dynamic_pointer_cast<opset::Clamp>(activation);
            a = clamp->get_max();
            b = clamp->get_min();
        } else {
            func = opset::ActivationFunction::IDENTITY;
        }

        std::shared_ptr<ngraph::Node> conv_activ;
        if (m_conv->get_input_size() == 2) {
            conv_activ = std::make_shared<opset::ArmConvolution>(
                    m_conv->input_value(Inputs::Data),
                    m_conv->input_value(Inputs::Weights),
                    m_conv->get_strides(),
                    m_conv->get_pads_begin(),
                    m_conv->get_pads_end(),
                    m_conv->get_dilations(),
                    m_conv->get_auto_pad(),
                    opset::ActivationInfo{func, a, b});
        } else {
            conv_activ = std::make_shared<opset::ArmConvolution>(
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
    });
}

template <typename Conv>
static auto addFuseBiasMatcher(pass::ConvBiasActivationFusion* pass) {
    auto conv = ngraph::pattern::wrap_type<Conv>();
    auto add = ngraph::pattern::wrap_type<opset::Add>({conv,  ngraph::pattern::any_input()});
    auto m = std::make_shared<ngraph::pattern::Matcher>(add, (std::string {Conv::type_info.name} + "BiasFusion"));

    pass->add_matcher(m, [] (ngraph::pattern::Matcher& m) {
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
    });
}

template <typename Conv, typename ArmConv>
static auto addConvertConvMatcher(pass::ConvBiasActivationFusion* pass) {
    auto conv = ngraph::pattern::wrap_type<Conv>();
    auto m = std::make_shared<ngraph::pattern::Matcher>(conv, (std::string {Conv::type_info.name} + "ConvertToArmConvolution"));

    pass->add_matcher(m, [] (ngraph::pattern::Matcher& m) {
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
    });
}

ArmPlugin::pass::ConvBiasActivationFusion::ConvBiasActivationFusion() : GraphRewrite() {
    // Convert Single Conv to ArmConv to reduce number of matchers below
    addConvertConvMatcher<opset::Convolution, opset::ArmConvolution>(this);
    addConvertConvMatcher<opset::GroupConvolution, opset::ArmGroupConvolution>(this);

    addFuseBiasMatcher<opset::ArmConvolution>(this);
    addFuseBiasMatcher<opset::ArmGroupConvolution>(this);

    addFuseActivationMatcher<opset::ArmConvolution, opset::Sigmoid>(this);
    addFuseActivationMatcher<opset::ArmConvolution, opset::Tanh>(this);
    addFuseActivationMatcher<opset::ArmConvolution, opset::Relu>(this);
    addFuseActivationMatcher<opset::ArmConvolution, opset::Clamp>(this);
    addFuseActivationMatcher<opset::ArmConvolution, opset::Abs>(this);
    addFuseActivationMatcher<opset::ArmConvolution, opset::Elu>(this);
    addFuseActivationMatcher<opset::ArmConvolution, opset::Sqrt>(this);
    addFuseActivationMatcher<opset::ArmConvolution, opset::SoftPlus>(this);
    addFuseActivationMatcher<opset::ArmConvolution, opset::HSwish>(this);
    addFuseActivationMatcher<opset::ArmConvolution, opset::PRelu>(this);

    addFuseActivationMatcher<opset::ArmGroupConvolution, opset::Sigmoid>(this);
    addFuseActivationMatcher<opset::ArmGroupConvolution, opset::Tanh>(this);
    addFuseActivationMatcher<opset::ArmGroupConvolution, opset::Relu>(this);
    addFuseActivationMatcher<opset::ArmGroupConvolution, opset::Clamp>(this);
    addFuseActivationMatcher<opset::ArmGroupConvolution, opset::Abs>(this);
    addFuseActivationMatcher<opset::ArmGroupConvolution, opset::Elu>(this);
    addFuseActivationMatcher<opset::ArmGroupConvolution, opset::Sqrt>(this);
    addFuseActivationMatcher<opset::ArmGroupConvolution, opset::SoftPlus>(this);
    addFuseActivationMatcher<opset::ArmGroupConvolution, opset::HSwish>(this);
    addFuseActivationMatcher<opset::ArmGroupConvolution, opset::PRelu>(this);
}
