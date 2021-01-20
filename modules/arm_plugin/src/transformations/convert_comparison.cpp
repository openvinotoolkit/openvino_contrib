// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <ngraph/rt_info.hpp>
#include <details/ie_exception.hpp>

#include "opset/opset.hpp"
#include "transformations/convert_comparison.hpp"


using namespace ArmPlugin;

template<typename T>
static auto addMatcher(ArmPlugin::pass::ConvertComparison* pass) {
    auto input0  = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 1});
    auto input1  = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 1});
    auto comparison = std::make_shared<T>(input0, input1);

    auto m = std::make_shared<ngraph::pattern::Matcher>(comparison, ("Convert" + std::string {T::type_info.name}).c_str());
    pass->add_matcher(m, [] (ngraph::pattern::Matcher& m) {
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
    });
}

ArmPlugin::pass::ConvertComparison::ConvertComparison() : GraphRewrite() {
    addMatcher<opset::Equal>(this);
    addMatcher<opset::NotEqual>(this);
    addMatcher<opset::Greater>(this);
    addMatcher<opset::GreaterEqual>(this);
    addMatcher<opset::Less>(this);
    addMatcher<opset::LessEqual>(this);
}
