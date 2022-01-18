// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <ngraph/rt_info.hpp>

#include "opset/opset.hpp"
#include "transformations/convert_comparison.hpp"

using namespace ArmPlugin;

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertComparisionBase, "ConvertComparisionBase", 0);
template <class T>
ngraph::matcher_pass_callback ArmPlugin::pass::ConvertComparisionBase::convert_comparision() {
    return [&](ngraph::pattern::Matcher& m) {
        auto comparison = m.get_match_root();
        if (!std::dynamic_pointer_cast<T>(comparison)) {
            return false;
        }

        std::vector<ngraph::Shape> shapes{comparison->get_input_shape(0), comparison->get_input_shape(1)};
        std::shared_ptr<ngraph::Node> comp_copy;
        if (shapes[0] != shapes[1]) {
            int brId;
            std::vector<int64_t> shape;
            if (shapes[0].size() != shapes[1].size()) {
                brId = shapes[0].size() < shapes[1].size() ? 0 : 1;
            } else {
                brId = ngraph::shape_size(shapes[0]) < ngraph::shape_size(shapes[1]) ? 0 : 1;
            }
            auto&& input = comparison->input_value(brId);
            auto targetShape = std::make_shared<opset::Constant>(ngraph::element::i64,
                                                                 ngraph::Shape{shapes[1 - brId].size()},
                                                                 std::vector<int64_t>(shapes[1 - brId].begin(), shapes[1 - brId].end()));
            auto broadcastedInp = std::make_shared<opset::Broadcast>(input, targetShape);
            ngraph::OutputVector inputs;
            if (brId == 0) {
                inputs = {broadcastedInp, comparison->input_value(1)};
            } else {
                inputs = {comparison->input_value(0), broadcastedInp};
            }
            comp_copy = comparison->clone_with_new_inputs(inputs);
        } else {
            comp_copy = comparison->clone_with_new_inputs(comparison->input_values());
        }

        auto out_shape = comparison->get_output_shape(0);
        auto total = ngraph::shape_size(out_shape);
        auto ones  = std::make_shared<opset::Constant>(ngraph::element::boolean, out_shape, std::vector<bool>(total, true));
        auto zeros = std::make_shared<opset::Constant>(ngraph::element::boolean, out_shape, std::vector<bool>(total, false));
        auto new_node = std::make_shared<opset::Select>(comp_copy, ones, zeros);

        new_node->set_friendly_name(comparison->get_friendly_name());
        ngraph::copy_runtime_info(comparison, new_node);
        ngraph::replace_node(comparison, new_node);
        return true;
    };
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertEqual, "ConvertEqual", 0);
ArmPlugin::pass::ConvertEqual::ConvertEqual() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::Equal>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                      ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                      ngraph::pattern::has_static_shape()), "ConvertEqual");
    register_matcher(m, convert_comparision<opset::Equal>());
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertNotEqual, "ConvertNotEqual", 0);
ArmPlugin::pass::ConvertNotEqual::ConvertNotEqual() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::NotEqual>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                         ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                         ngraph::pattern::has_static_shape()), "ConvertNotEqual");
    register_matcher(m, convert_comparision<opset::NotEqual>());
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertGreater, "ConvertGreater", 0);
ArmPlugin::pass::ConvertGreater::ConvertGreater() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::Greater>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                        ngraph::pattern::has_static_shape()), "ConvertGreater");
    register_matcher(m, convert_comparision<opset::Greater>());
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertGreaterEqual, "ConvertGreaterEqual", 0);
ArmPlugin::pass::ConvertGreaterEqual::ConvertGreaterEqual() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::GreaterEqual>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                             ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                             ngraph::pattern::has_static_shape()), "ConvertGreaterEqual");
    register_matcher(m, convert_comparision<opset::GreaterEqual>());
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertLess, "ConvertLess", 0);
ArmPlugin::pass::ConvertLess::ConvertLess() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::Less>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                     ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                     ngraph::pattern::has_static_shape()), "ConvertLess");
    register_matcher(m, convert_comparision<opset::Less>());
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertLessEqual, "ConvertLessEqual", 0);
ArmPlugin::pass::ConvertLessEqual::ConvertLessEqual() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::LessEqual>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                          ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                          ngraph::pattern::has_static_shape()), "ConvertLessEqual");
    register_matcher(m, convert_comparision<opset::LessEqual>());
}
