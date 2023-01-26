// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_pool_arm.hpp"
#include "transpose_utils.hpp"
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertArmMaxPoolV1, "ConvertArmMaxPoolV1", 0);
ArmPlugin::pass::ConvertArmMaxPoolV1::ConvertArmMaxPoolV1() {
    auto max_pool = ngraph::pattern::wrap_type<ov::op::v1::MaxPool>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto max_pool = std::dynamic_pointer_cast<ov::op::v1::MaxPool>(m.get_match_root());
        if (!max_pool) {
            return false;
        }
        if (std::dynamic_pointer_cast<opset::v1::ArmMaxPool>(m.get_match_root())) {
            return false;
        }

        size_t rank = max_pool->get_output_partial_shape(0).size();
        if (rank < 4 || rank > 5) {
            return false;
        }
        auto activations_transpose = transpose_on_input(max_pool->input_value(0), rank);
        auto output_shape = transpose_output_shape(max_pool, rank);
        auto arm_pool = std::make_shared<opset::v1::ArmMaxPool>(activations_transpose,
                                                              max_pool->get_strides(),
                                                              max_pool->get_pads_begin(),
                                                              max_pool->get_pads_end(),
                                                              max_pool->get_kernel(),
                                                              output_shape,
                                                              max_pool->get_rounding_type(),
                                                              max_pool->get_auto_pad());
        auto transpose = transpose_on_output(arm_pool, rank);
        transpose->set_friendly_name(max_pool->get_friendly_name());
        ngraph::copy_runtime_info(max_pool, {arm_pool, activations_transpose, transpose});
        ngraph::replace_node(max_pool, transpose);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(max_pool, "ConvertArmMaxPoolV1");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertArmMaxPoolV8, "ConvertArmMaxPoolV8", 0);
ArmPlugin::pass::ConvertArmMaxPoolV8::ConvertArmMaxPoolV8() {
    auto max_pool = ngraph::pattern::wrap_type<ov::op::v8::MaxPool>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto max_pool = std::dynamic_pointer_cast<ov::op::v8::MaxPool>(m.get_match_root());
        if (!max_pool) {
            return false;
        }
        if (std::dynamic_pointer_cast<opset::v8::ArmMaxPool>(m.get_match_root())) {
            return false;
        }

        size_t rank = max_pool->get_output_partial_shape(0).size();
        if (rank < 4 || rank > 5) {
            return false;
        }
        auto axis = max_pool->get_axis();
        if (axis > 1 || (axis < 0 && axis > -static_cast<int64_t>(rank) - 1)) {
            return false;
        }
        auto activations_transpose = transpose_on_input(max_pool->input_value(0), rank);
        auto output_shape = transpose_output_shape(max_pool, rank);
        auto arm_pool = std::make_shared<opset::v8::ArmMaxPool>(activations_transpose,
                                                              max_pool->get_strides(),
                                                              max_pool->get_dilations(),
                                                              max_pool->get_pads_begin(),
                                                              max_pool->get_pads_end(),
                                                              max_pool->get_kernel(),
                                                              output_shape,
                                                              max_pool->get_rounding_type(),
                                                              max_pool->get_auto_pad(),
                                                              max_pool->get_index_element_type(),
                                                              max_pool->get_axis());

        auto transpose = transpose_on_output(arm_pool->output(0), rank);
        transpose->set_friendly_name(max_pool->get_friendly_name() + ".0");
        auto transpose_on_indexes = transpose_on_output(arm_pool->output(1), rank);
        transpose_on_indexes->set_friendly_name(max_pool->get_friendly_name() + ".1");
        ngraph::copy_runtime_info(max_pool, {arm_pool, activations_transpose, transpose, transpose_on_indexes});
        ngraph::replace_node(max_pool, {transpose, transpose_on_indexes});

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(max_pool, "ConvertArmMaxPoolV8");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertArmAvgPool, "ConvertArmAvgPool", 0);
ArmPlugin::pass::ConvertArmAvgPool::ConvertArmAvgPool() {
    auto avg_pool = ngraph::pattern::wrap_type<ov::op::v1::AvgPool>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto avg_pool = std::dynamic_pointer_cast<ov::op::v1::AvgPool>(m.get_match_root());
        if (!avg_pool) {
            return false;
        }
        if (std::dynamic_pointer_cast<opset::v1::ArmAvgPool>(m.get_match_root())) {
            return false;
        }

        size_t rank = avg_pool->get_output_partial_shape(0).size();
        if (rank < 4 || rank > 5) {
            return false;
        }
        auto activations_transpose = transpose_on_input(avg_pool->input_value(0), rank);
        auto output_shape = transpose_output_shape(avg_pool, rank);
        auto arm_pool = std::make_shared<opset::v1::ArmAvgPool>(activations_transpose,
                                                          avg_pool->get_strides(),
                                                          avg_pool->get_pads_begin(),
                                                          avg_pool->get_pads_end(),
                                                          avg_pool->get_kernel(),
                                                          avg_pool->get_exclude_pad(),
                                                          output_shape,
                                                          avg_pool->get_rounding_type(),
                                                          avg_pool->get_auto_pad());
        auto transpose = transpose_on_output(arm_pool, rank);
        transpose->set_friendly_name(avg_pool->get_friendly_name());
        ngraph::copy_runtime_info(avg_pool, {arm_pool, activations_transpose, transpose});
        ngraph::replace_node(avg_pool, transpose);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(avg_pool, "ConvertArmAvgPool");
    register_matcher(m, callback);
}
