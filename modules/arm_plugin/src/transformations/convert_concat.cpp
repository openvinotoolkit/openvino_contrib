// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_concat.hpp"
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertConcat, "ConvertConcat", 0);
ArmPlugin::pass::ConvertConcat::ConvertConcat() {
    auto concat = ngraph::pattern::wrap_type<opset::Concat>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto concat = std::dynamic_pointer_cast<opset::Concat>(m.get_match_root());
        if (!concat) {
            return false;
        }

        auto src_type = concat->get_input_element_type(0);
        if (src_type != ngraph::element::f32 && src_type != ngraph::element::f16) {
            return false;
        }

        if (concat->get_shape().size() > 4) {
            return false;
        }

        auto arm_concat = std::make_shared<opset::ArmConcat>(concat->input_values(), concat->get_axis());
        arm_concat->set_friendly_name(concat->get_friendly_name());
        ngraph::copy_runtime_info(concat, arm_concat);
        ngraph::replace_node(concat, arm_concat);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat, "ConvertConcat");
    register_matcher(m, callback);
}
