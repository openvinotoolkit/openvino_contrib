// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_select.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::BroadcastSelect, "BroadcastSelect", 0);
ArmPlugin::pass::BroadcastSelect::BroadcastSelect() {
    auto select = ngraph::pattern::wrap_type<opset::Select>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto select = std::dynamic_pointer_cast<opset::Select>(m.get_match_root());
        if (!select) {
            return false;
        }

        bool isModified = false;
        auto out_shape = select->get_output_shape(0);
        auto shape = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{out_shape.size()}, out_shape);
        for (size_t i = 0; i < select->get_input_size(); i++) {
            if (select->get_input_shape(i) != out_shape) {
                auto broadcast = std::make_shared<opset::Broadcast>(select->input_value(i), shape);
                select->set_argument(i, broadcast);
                isModified = true;
            }
        }
        return isModified;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(select, "BroadcastSelect");
    register_matcher(m, callback);
}
