// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/decompose_mish.hpp"

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>

ArmPlugin::pass::DecomposeMish::DecomposeMish() {
    auto mish = std::make_shared<opset::Mish>(ngraph::pattern::any_input());

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto mish = std::dynamic_pointer_cast<opset::Mish>(m.get_match_root());
        if (!mish) {
            return false;
        }

        auto exp = std::make_shared<opset::Exp>(mish->input_value(0));
        auto add = std::make_shared<opset::Add>(exp, opset::Constant::create(mish->get_element_type(), ngraph::Shape{1}, {1.0f}));
        auto log = std::make_shared<opset::Log>(add);
        auto tanh = std::make_shared<opset::Tanh>(log);
        auto mul = std::make_shared<opset::Multiply>(mish->input_value(0), tanh);

        mul->set_friendly_name(mish->get_friendly_name());
        ngraph::copy_runtime_info(mish, {exp, add, log, tanh, mul});
        ngraph::replace_node(mish, mul);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mish, "DecomposeMish");
    register_matcher(m, callback);
}
