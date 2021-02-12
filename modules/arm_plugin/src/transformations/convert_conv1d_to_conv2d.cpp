// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_conv1d_to_conv2d.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

template <class Conv>
ngraph::matcher_pass_callback ArmPlugin::pass::ConvertConv1DBase::convert_conv1d_to_conv2d() {
    return [&](ngraph::pattern::Matcher& m) {
        auto conv = std::dynamic_pointer_cast<Conv>(m.get_match_root());
        if (!conv) {
            return false;
        }

        auto input_shape = conv->get_input_shape(0);
        // is Conv1D
        if (input_shape.size() != 3) {
            return false;
        }

        auto input   = conv->input_value(0);
        auto weights = conv->input_value(1);
        auto input2d_shape = input_shape;
        input2d_shape.push_back(1);
        auto in2d_shape = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{4}, input2d_shape);

        auto weights2d_shape = weights.get_shape();
        weights2d_shape.push_back(1);
        auto w_shape = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{weights2d_shape.size()}, weights2d_shape);

        auto input2d   = std::make_shared<opset::Reshape>(input, in2d_shape, true);
        auto weights2d = std::make_shared<opset::Reshape>(weights, w_shape, true);

        auto conv2d = std::make_shared<Conv>(input2d,
                                             weights2d,
                                             ngraph::Strides{conv->get_strides()[0], 1},
                                             ngraph::CoordinateDiff{conv->get_pads_begin()[0], 0},
                                             ngraph::CoordinateDiff{conv->get_pads_end()[0], 0},
                                             ngraph::Strides{conv->get_dilations()[0], 1},
                                             conv->get_auto_pad());

        auto in_shape = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{3}, conv->get_output_shape(0));
        auto reshape = std::make_shared<opset::Reshape>(conv2d, in_shape, true);

        reshape->set_friendly_name(conv->get_friendly_name());
        ngraph::copy_runtime_info(conv, {input2d, weights2d, conv2d, reshape});
        ngraph::replace_node(conv, reshape);
    };
}

ArmPlugin::pass::ConvertConv1D::ConvertConv1D() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<opset::Convolution>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                        ngraph::pattern::has_static_shape()), "ConvertConvolutionToArm");
    register_matcher(m, convert_conv1d_to_conv2d<opset::Convolution>());
}

ArmPlugin::pass::ConvertGroupConv1D::ConvertGroupConv1D() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::GroupConvolution>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                                 ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                                 ngraph::pattern::has_static_shape()), "ConvertGroupConvolutionToArm");
    register_matcher(m, convert_conv1d_to_conv2d<opset::GroupConvolution>());
}
