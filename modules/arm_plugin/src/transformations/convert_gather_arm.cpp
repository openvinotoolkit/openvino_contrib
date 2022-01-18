// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_gather_arm.hpp"
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertGather, "ConvertGather", 0);
ArmPlugin::pass::ConvertGather::ConvertGather() {
    auto gather = ngraph::pattern::wrap_type<opset::Gather>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto gather = std::dynamic_pointer_cast<opset::Gather>(m.get_match_root());
        if (!gather) {
            return false;
        }

        if (gather->get_input_shape(1).size() > 1) {
            return false;
        }

        auto axes = std::dynamic_pointer_cast<opset::Constant>(gather->input_value(2).get_node_shared_ptr());
        if (!axes) {
            return false;
        }

        auto arm_gather = std::make_shared<opset::ArmGather>(gather->input_value(0), gather->input_value(1), gather->input_value(2));
        arm_gather->set_friendly_name(gather->get_friendly_name());
        ngraph::copy_runtime_info(gather, arm_gather);
        ngraph::replace_node(gather, arm_gather);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gather, "ConvertGather");
    register_matcher(m, callback);
}
