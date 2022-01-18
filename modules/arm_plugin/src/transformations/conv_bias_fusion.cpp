// Copyright (C) 2020-2022 Intel Corporation
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
#include <ngraph_ops/type_relaxed.hpp>

using namespace ArmPlugin;

enum Layout {N, C, H, W};
enum Inputs {Data, Weights, Bias};

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvBiasFusionBase, "ConvBiasFusionBase", 0);
template <class Conv, class Eltwise>
void ArmPlugin::pass::ConvBiasFusionBase::registerMatcher(const std::string& name) {
    auto conv_pattern = ngraph::pattern::wrap_type<Conv>(ngraph::pattern::consumers_count(1));
    auto bias_pattern = ngraph::pattern::any_input();
    auto eltwise_pattern = ngraph::pattern::wrap_type<Eltwise>({conv_pattern, bias_pattern});
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(eltwise_pattern, name), [=](ngraph::pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto conv = std::static_pointer_cast<Conv>(pattern_map[conv_pattern].get_node_shared_ptr());
        auto bias = pattern_map[bias_pattern].get_node_shared_ptr();
        auto eltwise = pattern_map[eltwise_pattern].get_node_shared_ptr();

        if (!ngraph::is_type<opset::Constant>(bias.get())) {
            return false;
        }

        if (conv->output(0).get_target_inputs().size() != 1) {
            return false;
        }

        ngraph::Output<ngraph::Node> new_bias(bias);

        auto bias_shape = bias->get_output_shape(0);
        auto output_pshape = conv->get_output_partial_shape(0);
        const auto channel_dim = output_pshape[Layout::C].get_length();

        if (output_pshape.rank().is_dynamic() || output_pshape[Layout::C].is_dynamic()) {
            return false;
        }

        int broad_dims = conv->get_shape().size() - bias_shape.size();
        int channel_ind = Layout::C - broad_dims;
        if (channel_ind < 0 || bias_shape[channel_ind] != channel_dim || ngraph::shape_size(bias_shape) != channel_dim) {
            return false;
        }

        if (bias_shape.size() > 1) {
            new_bias = std::make_shared<opset::Reshape>(bias,
                    opset::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {channel_dim}), true);
        }

        if (conv->inputs().size() == 3) {
            new_bias = std::make_shared<opset::Add>(
                std::make_shared<opset::Convert>(new_bias, conv->get_input_element_type(Inputs::Bias)),
                conv->input_value(Inputs::Bias));
        }

        auto quantized = (conv->get_input_element_type(Inputs::Data).is_quantized() ||
                          conv->get_input_element_type(Inputs::Weights).is_quantized());

        // if (quantized) {
        //     new_bias = std::make_shared<opset::Convert>(new_bias, ngraph::element::i32);
        // }

        auto outputType = conv->get_output_element_type(0);
        IE_ASSERT(outputType.is_real());

        std::shared_ptr<ngraph::Node> new_conv;
        if (quantized) {
            new_conv = std::make_shared<ngraph::op::TypeRelaxed<Conv>>(
                std::vector<ngraph::element::Type>{outputType, outputType, outputType},
                std::vector<ngraph::element::Type>{outputType},
                ngraph::op::TemporaryReplaceOutputType(conv->input_value(Inputs::Data), outputType).get(),
                ngraph::op::TemporaryReplaceOutputType(conv->input_value(Inputs::Weights), outputType).get(),
                ngraph::op::TemporaryReplaceOutputType(new_bias, outputType).get(),
                conv->get_strides(),
                conv->get_pads_begin(),
                conv->get_pads_end(),
                conv->get_dilations(),
                conv->get_auto_pad());
        } else {
            new_conv = std::make_shared<Conv>(
                conv->input_value(Inputs::Data),
                conv->input_value(Inputs::Weights),
                new_bias,
                conv->get_strides(),
                conv->get_pads_begin(),
                conv->get_pads_end(),
                conv->get_dilations(),
                conv->get_auto_pad());
        }

        ngraph::copy_runtime_info({conv, eltwise}, new_conv);
        new_conv->set_friendly_name(eltwise->get_friendly_name());
        ngraph::replace_node(eltwise, new_conv);
        return true;
    });
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertConvBase, "ConvertConvBase", 0);
template <class Conv, class ArmConv>
ngraph::matcher_pass_callback ArmPlugin::pass::ConvertConvBase::convert_conv_to_arm_conv() {
    return [&](ngraph::pattern::Matcher& m) {
        auto conv = std::dynamic_pointer_cast<Conv>(m.get_match_root());
        if (!conv) {
            return false;
        }
        auto quantized = (conv->get_input_element_type(Inputs::Data).is_quantized() ||
                          conv->get_input_element_type(Inputs::Weights).is_quantized());

        auto outputType = conv->get_output_element_type(0);
        IE_ASSERT(outputType.is_real());

        std::shared_ptr<ngraph::Node> conv_arm;
        if (quantized) {
            conv_arm = std::make_shared<ngraph::op::TypeRelaxed<ArmConv>>(
                std::vector<ngraph::element::Type>{outputType, outputType},
                std::vector<ngraph::element::Type>{outputType},
                ngraph::op::TemporaryReplaceOutputType(conv->input_value(Inputs::Data), outputType).get(),
                ngraph::op::TemporaryReplaceOutputType(conv->input_value(Inputs::Weights), outputType).get(),
                conv->get_strides(),
                conv->get_pads_begin(),
                conv->get_pads_end(),
                conv->get_dilations(),
                conv->get_auto_pad());
        } else {
            conv_arm = std::make_shared<ArmConv>(
                conv->input_value(Inputs::Data),
                conv->input_value(Inputs::Weights),
                conv->get_strides(),
                conv->get_pads_begin(),
                conv->get_pads_end(),
                conv->get_dilations(),
                conv->get_auto_pad());
        }

        ngraph::copy_runtime_info(conv, conv_arm);
        conv_arm->set_friendly_name(conv->get_friendly_name());
        ngraph::replace_node(conv, conv_arm);
        return true;
    };
}

// ----------------------------------------ConvertConvolution----------------------------------------

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertSingleConvolutionToArm, "ConvertSingleConvolutionToArm", 0);
ArmPlugin::pass::ConvertSingleConvolutionToArm::ConvertSingleConvolutionToArm() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<opset::Convolution>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                        ngraph::pattern::has_static_shape()), "ConvertConvolutionToArm");
    register_matcher(m, convert_conv_to_arm_conv<opset::Convolution, opset::ArmConvolution>());
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertGroupConvolutionToArm, "ConvertGroupConvolutionToArm", 0);
ArmPlugin::pass::ConvertGroupConvolutionToArm::ConvertGroupConvolutionToArm() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::GroupConvolution>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                                 ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                                 ngraph::pattern::has_static_shape()), "ConvertGroupConvolutionToArm");
    register_matcher(m, convert_conv_to_arm_conv<opset::GroupConvolution, opset::ArmGroupConvolution>());
}

// ------------------------------------------ConvBiasFusion------------------------------------------
NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvAddFusion, "ConvAddFusion", 0);
ArmPlugin::pass::ConvAddFusion::ConvAddFusion() {
    registerMatcher<opset::ArmConvolution, opset::Add>("ConvAddFusion");
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::GroupConvAddFusion, "GroupConvAddFusion", 0);
ArmPlugin::pass::GroupConvAddFusion::GroupConvAddFusion() {
    registerMatcher<opset::ArmGroupConvolution, opset::Add>("GroupConvAddFusion");
}
