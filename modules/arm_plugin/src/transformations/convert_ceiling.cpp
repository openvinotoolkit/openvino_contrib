// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_ceiling.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertCeiling, "ConvertCeiling", 0);

ArmPlugin::pass::ConvertCeiling::ConvertCeiling() {
    auto ceil = ngraph::pattern::wrap_type<opset::Ceiling>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto ceil = std::dynamic_pointer_cast<opset::Ceiling>(m.get_match_root());
        if (!ceil) {
            return false;
        }

        auto input = ceil->input_value(0);
        auto floor = std::make_shared<opset::Floor>(input);
        auto greater = std::make_shared<opset::Greater>(input, floor);

        auto ones = std::make_shared<opset::Constant>(ceil->get_input_element_type(0), ngraph::Shape{1}, std::vector<float>{1});
        auto add = std::make_shared<opset::Add>(floor, ones);
        auto select = std::make_shared<opset::Select>(greater, add, input);

        select->set_friendly_name(ceil->get_friendly_name());
        ngraph::copy_runtime_info(ceil, {floor, greater, add, select});
        ngraph::replace_node(ceil, select);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(ceil, "ConvertCeiling");
    register_matcher(m, callback);
}
