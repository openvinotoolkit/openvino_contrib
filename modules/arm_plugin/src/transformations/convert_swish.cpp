// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_swish.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>

ArmPlugin::pass::ConvertSwish::ConvertSwish() : GraphRewrite() {
    {
        auto input = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 1});
        auto beta  = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{});
        auto swishBeta = std::make_shared<opset::Swish>(input, beta);
        add_matcher(std::make_shared<ngraph::pattern::Matcher>(swishBeta, "ConvertSwishWithBeta"),
            [](ngraph::pattern::Matcher& m) {
            // Swish(x, beta) = x * sigmoid(beta * x)
            auto swish = std::dynamic_pointer_cast<opset::Swish>(m.get_match_root());
            if (!swish) {
                return false;
            }

            auto beta = swish->input_value(1);

            auto shape = opset::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
            auto reshape = std::make_shared<opset::Reshape>(beta, shape, true);

            auto arg = std::make_shared<opset::Multiply>(swish->input_value(0), reshape);
            auto sigmoid = std::make_shared<opset::Sigmoid>(arg);
            auto mul = std::make_shared<opset::Multiply>(swish->input_value(0), sigmoid);

            mul->set_friendly_name(swish->get_friendly_name());
            ngraph::copy_runtime_info(swish, {arg, sigmoid, mul});
            ngraph::replace_node(swish, mul);
            return true;
        }, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
    }
    {
        auto input = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 1});
        auto swish = std::make_shared<opset::Swish>(input);
        add_matcher(std::make_shared<ngraph::pattern::Matcher>(swish, "ConvertSwish"),
            [](ngraph::pattern::Matcher& m) {
            // Swish(x, 1) = x * sigmoid(1 * x)
            auto swish = std::dynamic_pointer_cast<opset::Swish>(m.get_match_root());
            if (!swish) {
                return false;
            }

            auto beta = opset::Constant::create(swish->get_element_type(), ngraph::Shape{1, 1}, {1.0});

            auto shape = opset::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
            auto reshape = std::make_shared<opset::Reshape>(beta, shape, true);

            auto arg = std::make_shared<opset::Multiply>(swish->input_value(0), reshape);
            auto sigmoid = std::make_shared<opset::Sigmoid>(arg);
            auto mul = std::make_shared<opset::Multiply>(swish->input_value(0), sigmoid);

            mul->set_friendly_name(swish->get_friendly_name());
            ngraph::copy_runtime_info(swish, {arg, sigmoid, mul});
            ngraph::replace_node(swish, mul);
            return true;
        }, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
    }
}
