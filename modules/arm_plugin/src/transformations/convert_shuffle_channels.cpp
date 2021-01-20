// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_shuffle_channels.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>

ArmPlugin::pass::ConvertShuffleChannels::ConvertShuffleChannels() : GraphRewrite() {
    auto input = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 1});
    auto shuffle = std::make_shared<opset::ShuffleChannels>(input, 1, 1);

    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
        auto shuffle = std::dynamic_pointer_cast<opset::ShuffleChannels>(m.get_match_root());
        int axis = shuffle->get_axis();
        if (axis < 0) {
            axis += shuffle->get_input_shape(0).size();
        }

        if (!shuffle || axis == 1) {
            return false;
        }

        std::vector<int64_t> input_order(shuffle->get_input_shape(0).size());
        std::iota(input_order.begin(), input_order.end(), 0);
        std::swap(input_order[axis], input_order[1]);

        auto order = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);

        size_t group = shuffle->get_group();
        auto tr_forward = std::make_shared<opset::Transpose>(shuffle->input(0).get_source_output(), order);
        auto shuffle_new = std::make_shared<opset::ShuffleChannels>(tr_forward, 1, group);
        auto tr_backward = std::make_shared<opset::Transpose>(shuffle_new, order);

        tr_backward->set_friendly_name(shuffle->get_friendly_name());
        ngraph::copy_runtime_info(shuffle, {tr_forward, shuffle_new, tr_backward});
        ngraph::replace_node(shuffle, tr_backward);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(shuffle, "ConvertShuffleChannels");
    this->add_matcher(m, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}
