// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_ceiling.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>

ArmPlugin::pass::ConvertCeiling::ConvertCeiling() : GraphRewrite() {
    auto ceil = std::make_shared<opset::Ceiling>(ngraph::pattern::any_input());

    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
        auto ceil = std::dynamic_pointer_cast<opset::Ceiling>(m.get_match_root());
        if (!ceil) {
            return false;
        }

        auto input = ceil->input_value(0);
        auto floor = std::make_shared<opset::Floor>(input);
        auto greater = std::make_shared<opset::Greater>(input, floor);

        auto ones = std::make_shared<opset::Constant>(ceil->get_element_type(), ngraph::Shape{1}, std::vector<float>{1});
        auto add = std::make_shared<opset::Add>(floor, ones);
        auto select = std::make_shared<opset::Select>(greater, add, input);

        select->set_friendly_name(ceil->get_friendly_name());
        ngraph::copy_runtime_info(ceil, {floor, greater, add, select});
        ngraph::replace_node(ceil, select);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(ceil, "ConvertCeiling");
    this->add_matcher(m, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}
