// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/normalizel2_max_fusion.hpp"

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "transformations/utils/utils.hpp"

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::NormalizeL2Fusion, "NormalizeL2Fusion", 0);
ArmPlugin::pass::NormalizeL2Fusion::NormalizeL2Fusion() {
    auto input = ngraph::pattern::any_input();
    auto exp = ngraph::pattern::wrap_type<opset::Constant>();
    auto pow = std::make_shared<opset::Power>(input, exp);
    auto axes = ngraph::pattern::wrap_type<opset::Constant>();
    auto reduce_sum = std::make_shared<opset::ReduceSum>(pow, axes);
    auto eps_const = ngraph::pattern::wrap_type<opset::Constant>();
    auto max_eps = std::make_shared<opset::Maximum>(reduce_sum, eps_const);
    auto sqrt_max_eps = std::make_shared<opset::Sqrt>(max_eps);
    auto divide = std::make_shared<opset::Divide>(input, sqrt_max_eps);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();

        const auto data_input = pattern_to_output.at(input);
        const auto exp_input = std::dynamic_pointer_cast<opset::Constant>(pattern_to_output.at(exp).get_node_shared_ptr());
        const auto axes_input = std::dynamic_pointer_cast<opset::Constant>(pattern_to_output.at(axes).get_node_shared_ptr());
        const auto eps_attr = std::dynamic_pointer_cast<opset::Constant>(pattern_to_output.at(eps_const).get_node_shared_ptr());

        if (!exp_input || !axes_input || !eps_attr) {
            return false;
        }

        auto const_axes = axes_input->cast_vector<int64_t>();
        if (const_axes.size() != 1 || data_input.get_shape().size() - 1 - const_axes[0] > 2) {
            return false;
        }

        const bool is_square_pow = shape_size(exp_input->get_shape()) <= 1 && exp_input->cast_vector<int64_t>()[0] == 2;
        if (!is_square_pow) {
            return false;
        }
        if (shape_size(eps_attr->get_shape()) > 1) {
            return false;
        }
        const auto eps_attr_value = ngraph::op::util::has_constant_value<float>(exp_input, 2.0f);

        auto normalize_l2 = std::make_shared<opset::NormalizeL2>(data_input, axes_input, eps_attr_value, ngraph::op::EpsMode::MAX);

        normalize_l2->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(pow).get_node_shared_ptr(),
                                   pattern_to_output.at(reduce_sum).get_node_shared_ptr(),
                                   pattern_to_output.at(max_eps).get_node_shared_ptr(),
                                   pattern_to_output.at(sqrt_max_eps).get_node_shared_ptr(),
                                   pattern_to_output.at(divide).get_node_shared_ptr()
                                   },
                                   normalize_l2);
        ngraph::replace_node(m.get_match_root(), normalize_l2);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(divide, "NormalizeL2Fusion");
    register_matcher(m, callback);
}
