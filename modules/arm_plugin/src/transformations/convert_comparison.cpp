// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <ngraph/rt_info.hpp>
#include <details/ie_exception.hpp>

#include "opset/opset.hpp"
#include "transformations/convert_comparison.hpp"

using namespace ArmPlugin;

template <class T>
ngraph::matcher_pass_callback ArmPlugin::pass::ConvertComparisionBase::convert_comparision() {
    return [&](ngraph::pattern::Matcher& m) {
        auto comparison = m.get_match_root();
        if (!std::dynamic_pointer_cast<T>(comparison)) {
            return false;
        }

        if (comparison->get_input_shape(0) != comparison->get_input_shape(1)) {
            THROW_IE_EXCEPTION << std::string(T::type_info.name) + " op doesn't support broadcast";
        }

        auto out_shape = comparison->get_output_shape(0);
        auto total = ngraph::shape_size(out_shape);
        auto node_copy = comparison->clone_with_new_inputs(comparison->input_values());

        auto ones = std::make_shared<opset::Constant>(ngraph::element::u8, out_shape, std::vector<std::uint8_t>(total, 1));
        auto zeros = std::make_shared<opset::Constant>(ngraph::element::u8, out_shape, std::vector<std::uint8_t>(total, 0));
        auto new_node = std::make_shared<opset::Select>(node_copy, ones, zeros);

        new_node->set_friendly_name(comparison->get_friendly_name());
        ngraph::copy_runtime_info(comparison, new_node);
        ngraph::replace_node(comparison, new_node);
        return true;
    };
}

ArmPlugin::pass::ConvertEqual::ConvertEqual() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::Equal>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                      ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                      ngraph::pattern::has_static_shape()), "ConvertEqual");
    register_matcher(m, convert_comparision<opset::Equal>());
}

ArmPlugin::pass::ConvertNotEqual::ConvertNotEqual() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::NotEqual>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                         ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                         ngraph::pattern::has_static_shape()), "ConvertNotEqual");
    register_matcher(m, convert_comparision<opset::NotEqual>());
}

ArmPlugin::pass::ConvertGreater::ConvertGreater() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::Greater>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                        ngraph::pattern::has_static_shape()), "ConvertGreater");
    register_matcher(m, convert_comparision<opset::Greater>());
}

ArmPlugin::pass::ConvertGreaterEqual::ConvertGreaterEqual() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::GreaterEqual>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                             ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                             ngraph::pattern::has_static_shape()), "ConvertGreaterEqual");
    register_matcher(m, convert_comparision<opset::GreaterEqual>());
}

ArmPlugin::pass::ConvertLess::ConvertLess() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::Less>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                     ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                     ngraph::pattern::has_static_shape()), "ConvertLess");
    register_matcher(m, convert_comparision<opset::Less>());
}

ArmPlugin::pass::ConvertLessEqual::ConvertLessEqual() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::LessEqual>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                          ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                          ngraph::pattern::has_static_shape()), "ConvertLessEqual");
    register_matcher(m, convert_comparision<opset::LessEqual>());
}
