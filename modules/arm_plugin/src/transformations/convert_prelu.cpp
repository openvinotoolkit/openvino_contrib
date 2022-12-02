// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "convert_prelu.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::BroadcastPRelu, "BroadcastPRelu", 0);
ArmPlugin::pass::BroadcastPRelu::BroadcastPRelu() {
    auto prelu = ngraph::pattern::wrap_type<opset::PRelu>();

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
        auto slope_shape = prelu->get_input_shape(1);
        uint64_t constant_shape;

        std::vector<int64_t> broadcasted_shape;
        if (slope_shape.size() == 1 && input_shape[1] == slope.get_shape()[0]) {
            broadcasted_shape.assign(input_shape.size(), 1);
            broadcasted_shape[1] = slope.get_shape()[0]; // ChannelPRelu
            constant_shape = input_shape.size();
        } else {
            std::copy(slope.get_shape().begin(), slope.get_shape().end(), std::back_inserter(broadcasted_shape));
            constant_shape = slope_shape.size();       
        }

        auto shape   = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{constant_shape}, broadcasted_shape);
        auto reshape = std::make_shared<opset::Reshape>(slope, shape, true);

        prelu->set_argument(1, reshape);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prelu, "BroadcastPRelu");
    register_matcher(m, callback);
}
