// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <ngraph/rt_info.hpp>
#include <details/ie_exception.hpp>

#include "opset/opset.hpp"
#include "transformations/convert_logical.hpp"


using namespace ArmPlugin;

template<typename T>
static auto addMatcher(ArmPlugin::pass::ConvertLogical* pass) {
    auto binary_logical = std::make_shared<T>(ngraph::pattern::any_input(), ngraph::pattern::any_input());

    auto m = std::make_shared<ngraph::pattern::Matcher>(binary_logical, ("Convert" + std::string {T::type_info.name}).c_str());
    pass->add_matcher(m, [] (ngraph::pattern::Matcher& m) {
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
    });
}

template<> auto addMatcher<opset::LogicalNot>(ArmPlugin::pass::ConvertLogical* pass) {
    auto unary_logical = std::make_shared<opset::LogicalNot>(ngraph::pattern::any_input());

    auto m = std::make_shared<ngraph::pattern::Matcher>(unary_logical, "ConvertLogicalNot");
    pass->add_matcher(m, [] (ngraph::pattern::Matcher& m) {
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
    });
}

ArmPlugin::pass::ConvertLogical::ConvertLogical() : GraphRewrite() {
    addMatcher<opset::LogicalNot>(this);
    addMatcher<opset::LogicalAnd>(this);
    addMatcher<opset::LogicalOr>(this);
    addMatcher<opset::LogicalXor>(this);
}
