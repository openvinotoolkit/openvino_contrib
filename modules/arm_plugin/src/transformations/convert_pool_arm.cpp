// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_pool_arm.hpp"
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ArmPlugin::pass::ConvertArmMaxPoolV1::ConvertArmMaxPoolV1() {
    auto max_pool = ngraph::pattern::wrap_type<ov::op::v1::MaxPool>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto max_pool = std::dynamic_pointer_cast<ov::op::v1::MaxPool>(m.get_match_root());
        if (!max_pool) {
            return false;
        }

        auto arm_pool = std::make_shared<opset::v1::ArmMaxPool>(max_pool->input_value(0),
                                                              max_pool->get_strides(),
                                                              max_pool->get_pads_begin(),
                                                              max_pool->get_pads_end(),
                                                              max_pool->get_kernel(),
                                                              max_pool->get_rounding_type(),
                                                              max_pool->get_auto_pad());
        arm_pool->set_friendly_name(max_pool->get_friendly_name());
        ngraph::copy_runtime_info(max_pool, arm_pool);
        ngraph::replace_node(max_pool, arm_pool);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(max_pool, "ConvertArmMaxPoolV1");
    register_matcher(m, callback);
}

ArmPlugin::pass::ConvertArmMaxPoolV8::ConvertArmMaxPoolV8() {
    auto max_pool = ngraph::pattern::wrap_type<ov::op::v8::MaxPool>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto max_pool = std::dynamic_pointer_cast<ov::op::v8::MaxPool>(m.get_match_root());
        if (!max_pool) {
            return false;
        }

        auto arm_pool = std::make_shared<opset::v8::ArmMaxPool>(max_pool->input_value(0),
                                                              max_pool->get_strides(),
                                                              max_pool->get_dilations(),
                                                              max_pool->get_pads_begin(),
                                                              max_pool->get_pads_end(),
                                                              max_pool->get_kernel(),
                                                              max_pool->get_rounding_type(),
                                                              max_pool->get_auto_pad(),
                                                              max_pool->get_index_element_type(),
                                                              max_pool->get_axis());
        arm_pool->set_friendly_name(max_pool->get_friendly_name());
        ngraph::copy_runtime_info(max_pool, arm_pool);
        ngraph::replace_node(max_pool, arm_pool);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(max_pool, "ConvertArmMaxPoolV8");
    register_matcher(m, callback);
}

ArmPlugin::pass::ConvertArmAvgPool::ConvertArmAvgPool() {
    auto avg_pool = ngraph::pattern::wrap_type<ov::op::v1::AvgPool>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto avg_pool = std::dynamic_pointer_cast<ov::op::v1::AvgPool>(m.get_match_root());
        if (!avg_pool) {
            return false;
        }

        auto arm_pool = std::make_shared<opset::v1::ArmAvgPool>(avg_pool->input_value(0),
                                                          avg_pool->get_strides(),
                                                          avg_pool->get_pads_begin(),
                                                          avg_pool->get_pads_end(),
                                                          avg_pool->get_kernel(),
                                                          avg_pool->get_exclude_pad(),
                                                          avg_pool->get_rounding_type(),
                                                          avg_pool->get_auto_pad());
        arm_pool->set_friendly_name(avg_pool->get_friendly_name());
        ngraph::copy_runtime_info(avg_pool, arm_pool);
        ngraph::replace_node(avg_pool, arm_pool);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(avg_pool, "ConvertArmAvgPool");
    register_matcher(m, callback);
}