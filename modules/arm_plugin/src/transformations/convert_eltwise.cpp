// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_eltwise.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>

using namespace ArmPlugin;

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertEltwiseBase, "ConvertEltwiseBase", 0);
template <class T>
ngraph::matcher_pass_callback ArmPlugin::pass::ConvertEltwiseBase::convert_eltwise() {
    return [&](ngraph::pattern::Matcher& m) {
        auto eltwise = m.get_match_root();
        if (!std::dynamic_pointer_cast<T>(eltwise)) {
            return false;
        }

        if (eltwise->get_input_shape(0).size() != eltwise->get_input_shape(1).size()) {
            int broadcastedId = eltwise->get_input_shape(0).size() < eltwise->get_shape().size() ? 0 : 1;
            auto&& broadcastedInput = eltwise->input_value(broadcastedId);
            std::vector<int64_t> shape(eltwise->get_shape().size() - broadcastedInput.get_shape().size(), 1);
            std::copy(broadcastedInput.get_shape().begin(), broadcastedInput.get_shape().end(), std::back_inserter(shape));

            auto shape_node = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{shape.size()}, shape.data());
            auto reshape    = std::make_shared<opset::Reshape>(broadcastedInput, shape_node, true);

            eltwise->set_argument(broadcastedId, reshape);
            return true;
        }
        return false;
    };
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertAdd, "ConvertAdd", 0);
ArmPlugin::pass::ConvertAdd::ConvertAdd() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::Add>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                    ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                    ngraph::pattern::has_static_shape()), "ConvertAdd");
    register_matcher(m, convert_eltwise<opset::Add>());
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertSubtract, "ConvertSubtract", 0);
ArmPlugin::pass::ConvertSubtract::ConvertSubtract() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::Subtract>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                         ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                         ngraph::pattern::has_static_shape()), "ConvertSubtract");
    register_matcher(m, convert_eltwise<opset::Subtract>());
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertMultiply, "ConvertMultiply", 0);
ArmPlugin::pass::ConvertMultiply::ConvertMultiply() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::Multiply>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                         ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                         ngraph::pattern::has_static_shape()), "ConvertMultiply");
    register_matcher(m, convert_eltwise<opset::Multiply>());
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertMinimum, "ConvertMinimum", 0);
ArmPlugin::pass::ConvertMinimum::ConvertMinimum() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::Minimum>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                        ngraph::pattern::has_static_shape()), "ConvertMinimum");
    register_matcher(m, convert_eltwise<opset::Minimum>());
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertMaximum, "ConvertMaximum", 0);
ArmPlugin::pass::ConvertMaximum::ConvertMaximum() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::Maximum>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                        ngraph::pattern::has_static_shape()), "ConvertMaximum");
    register_matcher(m, convert_eltwise<opset::Maximum>());
}
