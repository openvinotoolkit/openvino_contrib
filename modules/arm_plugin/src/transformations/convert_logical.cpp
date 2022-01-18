// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <ngraph/rt_info.hpp>

#include "opset/opset.hpp"
#include "transformations/convert_logical.hpp"


using namespace ArmPlugin;

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertLogicalBase, "ConvertLogicalBase", 0);
template <class T>
ngraph::matcher_pass_callback ArmPlugin::pass::ConvertLogicalBase::convert_logical() {
    return [&](ngraph::pattern::Matcher& m) {
        auto binary_logical = m.get_match_root();
        if (!std::dynamic_pointer_cast<T>(binary_logical)) {
            return false;
        }

        auto out_shape = binary_logical->get_output_shape(0);
        auto total = ngraph::shape_size(out_shape);
        auto false_node = std::make_shared<opset::Constant>(ngraph::element::boolean, out_shape, std::vector<bool>(total, false));
        auto true_node = std::make_shared<opset::Constant>(ngraph::element::boolean, out_shape, std::vector<bool>(total, true));

        std::shared_ptr<ngraph::Node> logical;
        if (std::is_same<T, opset::LogicalOr>()) {
            logical = std::make_shared<opset::Select>(binary_logical->input_value(0), true_node, binary_logical->input_value(1));
        } else if (std::is_same<T, opset::LogicalAnd>()) {
            logical = std::make_shared<opset::Select>(binary_logical->input_value(0), binary_logical->input_value(1), false_node);
        } else if (std::is_same<T, opset::LogicalXor>()) {
            auto not_second = std::make_shared<opset::Select>(binary_logical->input_value(1), false_node, true_node);
            logical = std::make_shared<opset::Select>(binary_logical->input_value(0), not_second, binary_logical->input_value(1));
        }

        logical->set_friendly_name(binary_logical->get_friendly_name());
        ngraph::copy_runtime_info(binary_logical, logical);
        ngraph::replace_node(binary_logical, logical);
        return true;
    };
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertLogicalNot, "ConvertLogicalNot", 0);
ArmPlugin::pass::ConvertLogicalNot::ConvertLogicalNot() {
    auto logical_not = std::make_shared<opset::LogicalNot>(ngraph::pattern::any_input());

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto logical_not = std::dynamic_pointer_cast<opset::LogicalNot>(m.get_match_root());
        auto unary_logical = m.get_match_root();
        if (!std::dynamic_pointer_cast<opset::LogicalNot>(unary_logical)) {
            return false;
        }

        auto out_shape = unary_logical->get_output_shape(0);
        auto total = ngraph::shape_size(out_shape);
        auto false_node = std::make_shared<opset::Constant>(ngraph::element::boolean, out_shape, std::vector<bool>(total, false));
        auto true_node = std::make_shared<opset::Constant>(ngraph::element::boolean, out_shape, std::vector<bool>(total, true));
        auto logical = std::make_shared<opset::Select>(unary_logical->input_value(0), false_node, true_node);

        logical->set_friendly_name(unary_logical->get_friendly_name());
        ngraph::copy_runtime_info(unary_logical, logical);
        ngraph::replace_node(unary_logical, logical);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(logical_not, "ConvertLogicalNot");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertLogicalAnd, "ConvertLogicalAnd", 0);
ArmPlugin::pass::ConvertLogicalAnd::ConvertLogicalAnd() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::LogicalAnd>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                           ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                           ngraph::pattern::has_static_shape()), "ConvertLogicalAnd");
    register_matcher(m, convert_logical<opset::LogicalAnd>());
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertLogicalOr, "ConvertLogicalOr", 0);
ArmPlugin::pass::ConvertLogicalOr::ConvertLogicalOr() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::LogicalOr>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                          ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                          ngraph::pattern::has_static_shape()), "ConvertLogicalOr");
    register_matcher(m, convert_logical<opset::LogicalOr>());
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertLogicalXor, "ConvertLogicalXor", 0);
ArmPlugin::pass::ConvertLogicalXor::ConvertLogicalXor() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::LogicalXor>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                           ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                           ngraph::pattern::has_static_shape()), "ConvertLogicalXor");
    register_matcher(m, convert_logical<opset::LogicalXor>());
}
