// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_round.hpp"
#include "opset/opset.hpp"

#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertRound, "ConvertRound", 0);
ArmPlugin::pass::ConvertRound::ConvertRound() {
    auto round = ngraph::pattern::wrap_type<opset::Round>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto round = std::dynamic_pointer_cast<opset::Round>(m.get_match_root());

        if (!round) {
            return false;
        }

        auto mode  = round->get_mode();
        if (mode != ngraph::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO) {
            return false;
        }

        auto input = round->input_value(0);
        auto abs_input = std::make_shared<opset::Abs>(input);
        auto ceil  = std::make_shared<opset::Ceiling>(abs_input);
        auto floor = std::make_shared<opset::Floor>(abs_input);
        auto floor_diff = std::make_shared<opset::Subtract>(abs_input, floor);

        auto out_shape = round->get_output_shape(0);
        auto total = ngraph::shape_size(out_shape);
        auto thr   = std::make_shared<opset::Constant>(round->get_input_element_type(0), out_shape, std::vector<float>(total, 0.5f));

        auto mask = std::make_shared<opset::Less>(floor_diff, thr);
        auto result_positive = std::make_shared<opset::Select>(mask, floor, ceil);
        auto sign = std::make_shared<opset::Sign>(input);
        auto result = std::make_shared<opset::Multiply>(sign, result_positive);

        result->set_friendly_name(round->get_friendly_name());
        ngraph::copy_runtime_info(round, {abs_input, ceil, floor, floor_diff, mask, sign, result});
        ngraph::replace_node(round, result);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(round, "ConvertRound");
    register_matcher(m, callback);
}