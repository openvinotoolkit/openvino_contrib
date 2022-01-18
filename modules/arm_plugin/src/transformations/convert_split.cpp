// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_split.hpp"
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertSplit, "ConvertSplit", 0);
ArmPlugin::pass::ConvertSplit::ConvertSplit() {
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(ngraph::pattern::wrap_type<opset::Split>(), "ConvertSplit"),
        [](ngraph::pattern::Matcher& m) {
            auto split = std::dynamic_pointer_cast<opset::Split>(m.get_match_root());
            if (!split) {
                return false;
            }
            if (split->get_input_shape(0).size() > 4) {
                return false;
            }

            auto arm_split = std::make_shared<opset::ArmSplit>(split->input_value(0), split->input_value(1), split->get_num_splits());
            arm_split->set_friendly_name(split->get_friendly_name());
            ngraph::copy_runtime_info(split, arm_split);
            ngraph::replace_node(split, arm_split);
            return true;
        });
}
