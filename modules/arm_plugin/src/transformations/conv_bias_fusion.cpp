// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <details/ie_exception.hpp>
#include "conv_bias_fusion.hpp"

#include <memory>
#include <numeric>
#include <vector>

#include <ngraph/rt_info.hpp>

#include "opset/opset.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>

using namespace ArmPlugin;

enum Layout {N, C, H, W};
enum Inputs {Data, Weights, Bias};

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

        if (m_conv->output(0).get_target_inputs().size() != 1) {
            return false;
        }

        if (!std::dynamic_pointer_cast<opset::Constant>(eltwise->input_value(1 - conv_idx).get_node_shared_ptr())) {
            return false; // Unsupported Convolution with inconstant bias
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
                    m_conv->get_auto_pad());

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

        if (!std::dynamic_pointer_cast<opset::Constant>(m_conv->input_value(Inputs::Weights).get_node_shared_ptr())) {
            IE_THROW() << "Unsupported Convolution with inconstant weights.";
        }

        auto conv_arm = std::make_shared<ArmConv>(
                    m_conv->input_value(Inputs::Data),
                    m_conv->input_value(Inputs::Weights),
                    m_conv->get_strides(),
                    m_conv->get_pads_begin(),
                    m_conv->get_pads_end(),
                    m_conv->get_dilations(),
                    m_conv->get_auto_pad());

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
            ngraph::pattern::wrap_type<opset::Add>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                    ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                    ngraph::pattern::has_static_shape()), "ConvBiasFusion");
    register_matcher(m, fuse_conv_with_bias<opset::ArmConvolution>());
}

ArmPlugin::pass::GroupConvBiasFusion::GroupConvBiasFusion() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                ngraph::pattern::wrap_type<opset::Add>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                        ngraph::pattern::has_static_shape()), "GroupConvBiasFusion");
    register_matcher(m, fuse_conv_with_bias<opset::ArmGroupConvolution>());
}