// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_gather_arm.hpp"
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>


ArmPlugin::pass::ConvertGather::ConvertGather() {
    auto gather = ngraph::pattern::wrap_type<opset::Gather>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        std::cout << "1" << std::endl;
        auto gather = std::dynamic_pointer_cast<opset::Gather>(m.get_match_root());
        std::cout << "2" << std::endl;
        if (!gather) {
            return false;
        }

        std::cout << "3" << std::endl;
        if (gather->get_input_shape(1).size() > 1) {
            return false;
        }

        std::cout << "4" << std::endl;
        auto axes = std::dynamic_pointer_cast<opset::Constant>(gather->input_value(2).get_node_shared_ptr());
        std::cout << "5" << std::endl;
        if (!axes) {
            return false;
        }
        std::cout << "6" << std::endl;

        auto arm_gather = std::make_shared<opset::ArmGather>(gather->input_value(0), gather->input_value(1), gather->input_value(2));
        std::cout << "7" << std::endl;
        arm_gather->set_friendly_name(gather->get_friendly_name());
        std::cout << "8" << std::endl;
        ngraph::copy_runtime_info(gather, arm_gather);
        std::cout << "9" << std::endl;
        ngraph::replace_node(gather, arm_gather);
        std::cout << "10" << std::endl;
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gather, "ConvertGather");
    register_matcher(m, callback);
}
