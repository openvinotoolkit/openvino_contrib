// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_eltwise.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>

using namespace ArmPlugin;

template<typename T>
static auto addMatcher(ArmPlugin::pass::ConvertEltwise* pass) {
    auto input0  = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 1});
    auto input1  = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 1});
    auto eltwise = std::make_shared<T>(input0, input1);

    auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise, ("Convert" + std::string {T::type_info.name} + "Scalar").c_str());
    pass->add_matcher(m, [] (ngraph::pattern::Matcher& m) {
        auto eltwise = m.get_match_root();
        if (!std::dynamic_pointer_cast<T>(eltwise)) {
            return false;
        }

        if (eltwise->get_input_shape(0).size() != eltwise->get_input_shape(1).size()) {
            int broadcastedId = eltwise->get_input_shape(0).size() < eltwise->get_shape().size() ? 0 : 1;
            auto&& broadcastedInput = eltwise->input(broadcastedId);
            std::vector<int64_t> shape(eltwise->get_shape().size() - broadcastedInput.get_shape().size(), 1);
            std::copy(broadcastedInput.get_shape().begin(), broadcastedInput.get_shape().end(), std::back_inserter(shape));

            auto shape_node = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{shape.size()}, shape.data());
            auto reshape    = std::make_shared<opset::Reshape>(broadcastedInput.get_source_output(), shape_node, true);

            eltwise->set_argument(broadcastedId, reshape);
            return true;
        }
        return false;
    });
}

ArmPlugin::pass::ConvertEltwise::ConvertEltwise() : GraphRewrite() {
    addMatcher<opset::Add>(this);
    addMatcher<opset::Subtract>(this);
    addMatcher<opset::Multiply>(this);
    addMatcher<opset::Minimum>(this);
    addMatcher<opset::Maximum>(this);
}
