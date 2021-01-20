// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_sign.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>

ArmPlugin::pass::ConvertSign::ConvertSign() : GraphRewrite() {
    auto input = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::i64, ngraph::Shape{1, 1, 1, 1});
    auto sign = std::make_shared<opset::Sign>(input);

    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
        auto sign = std::dynamic_pointer_cast<opset::Sign>(m.get_match_root());

        if (!sign) {
            return false;
        }

        auto out_shape = sign->get_output_shape(0);
        auto type = sign->input_value(0).get_element_type();
        auto total = ngraph::shape_size(out_shape);
        auto positive = std::make_shared<opset::Constant>(type, out_shape, std::vector<int>(total, 1));
        auto negative = std::make_shared<opset::Constant>(type, out_shape, std::vector<int>(total, -1));
        auto zeros    = std::make_shared<opset::Constant>(type, out_shape, std::vector<int>(total, 0));

        auto less = std::make_shared<opset::Less>(sign->input_value(0), zeros);
        auto is_negative = std::make_shared<opset::Select>(less, negative, positive);

        auto equal = std::make_shared<opset::Equal>(sign->input_value(0), zeros);
        auto result = std::make_shared<opset::Select>(equal, zeros, is_negative);

        result->set_friendly_name(sign->get_friendly_name());
        ngraph::copy_runtime_info(sign, {less, is_negative, equal, result});
        ngraph::replace_node(sign, result);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(sign, "ConvertSign");
    this->add_matcher(m, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}
