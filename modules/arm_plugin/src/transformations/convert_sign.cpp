// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_sign.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertSign, "ConvertSign", 0);
ArmPlugin::pass::ConvertSign::ConvertSign() {
    auto sign = ngraph::pattern::wrap_type<opset::Sign>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto sign = std::dynamic_pointer_cast<opset::Sign>(m.get_match_root());

        if (!sign) {
            return false;
        }

        auto out_shape = sign->get_output_shape(0);
        auto type = sign->get_input_element_type(0);
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
    register_matcher(m, callback);
}
