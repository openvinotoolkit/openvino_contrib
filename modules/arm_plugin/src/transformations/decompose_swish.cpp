// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/decompose_swish.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::DecomposeSingleSwish, "DecomposeSingleSwish", 0);
ArmPlugin::pass::DecomposeSingleSwish::DecomposeSingleSwish() {
    auto swish = ngraph::pattern::wrap_type<opset::Swish>({ngraph::pattern::any_input()});

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        // Swish(x, 1) = x * sigmoid(1 * x)
        auto swish = std::dynamic_pointer_cast<opset::Swish>(m.get_match_root());
        if (!swish) {
            return false;
        }

        auto beta = opset::Constant::create(swish->get_output_element_type(0), ngraph::Shape{1, 1}, {1.0});

        auto shape = opset::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto reshape = std::make_shared<opset::Reshape>(beta, shape, true);

        auto arg = std::make_shared<opset::Multiply>(swish->input_value(0), reshape);
        auto sigmoid = std::make_shared<opset::Sigmoid>(arg);
        auto mul = std::make_shared<opset::Multiply>(swish->input_value(0), sigmoid);

        mul->set_friendly_name(swish->get_friendly_name());
        ngraph::copy_runtime_info(swish, {arg, sigmoid, mul});
        ngraph::replace_node(swish, mul);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(swish, "DecomposeSingleSwish");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::DecomposeSwishWithBeta, "DecomposeSwishWithBeta", 0);
ArmPlugin::pass::DecomposeSwishWithBeta::DecomposeSwishWithBeta() {
    auto swish = ngraph::pattern::wrap_type<opset::Swish>({ngraph::pattern::any_input(), ngraph::pattern::any_input()});

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
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
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(swish, "DecomposeSwishWithBeta");
    register_matcher(m, callback);
}
