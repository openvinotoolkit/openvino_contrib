// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_prelu.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>

ArmPlugin::pass::BroadcastPRelu::BroadcastPRelu() {
    auto prelu = std::make_shared<opset::PRelu>(ngraph::pattern::any_input(), ngraph::pattern::any_input());

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto prelu = std::dynamic_pointer_cast<opset::PRelu>(m.get_match_root());
        if (!prelu) {
            return false;
        }

        auto input_shape = prelu->get_input_shape(0);
        if (input_shape.size() == 1) {
            return false;
        }

        auto input = prelu->input_value(0);
        auto slope = prelu->input_value(1);
        std::vector<int64_t> broadcasted_shape(input_shape.size(), 1);
        broadcasted_shape[1] = slope.get_shape()[0]; // ChannelPRelu

        auto shape   = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{input_shape.size()}, broadcasted_shape);
        auto reshape = std::make_shared<opset::Reshape>(slope, shape, true);

        prelu->set_argument(1, reshape);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prelu, "BroadcastPRelu");
    register_matcher(m, callback);
}
