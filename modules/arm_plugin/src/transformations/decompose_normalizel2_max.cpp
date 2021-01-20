// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/decompose_normalizel2_max.hpp"

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>

ArmPlugin::pass::DecomposeNormalizeL2Max::DecomposeNormalizeL2Max() : GraphRewrite() {
    auto input = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 1});
    auto axes  = opset::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
    auto normalize_l2 = std::make_shared<opset::NormalizeL2>(input, axes, 1.0f, ngraph::op::EpsMode::MAX);

    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
        auto normalize_l2 = std::dynamic_pointer_cast<opset::NormalizeL2>(m.get_match_root());
        auto input = normalize_l2->input_value(0).get_node_shared_ptr();
        auto axes = normalize_l2->input_value(1).get_node_shared_ptr();

        if (normalize_l2->get_eps_mode() != ngraph::op::EpsMode::MAX) {
            return false;
        }

        const auto eps_attr = normalize_l2->get_eps();
        if (!axes || !eps_attr) {
            return false;
        }

        auto exp = opset::Constant::create(input->get_element_type(), ngraph::Shape{1}, {2.0f});
        auto pow = std::make_shared<opset::Power>(input, exp);
        auto reduce_sum = std::make_shared<opset::ReduceSum>(pow, axes, true);
        auto sqrt = std::make_shared<opset::Sqrt>(reduce_sum);
        auto eps = opset::Constant::create(input->get_element_type(), ngraph::Shape{1}, {eps_attr});
        auto sqrt_max_eps = std::make_shared<opset::Maximum>(sqrt, eps);
        auto divide = std::make_shared<opset::Divide>(input, sqrt_max_eps);

        divide->set_friendly_name(normalize_l2->get_friendly_name());
        ngraph::copy_runtime_info(normalize_l2, {pow, reduce_sum, sqrt, sqrt_max_eps, divide});
        ngraph::replace_node(normalize_l2, divide);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(normalize_l2, "DecomposeNormalizeL2Max");
    this->add_matcher(m, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}
