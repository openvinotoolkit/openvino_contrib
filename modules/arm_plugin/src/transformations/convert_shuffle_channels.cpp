// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_shuffle_channels.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertShuffleChannels, "ConvertShuffleChannels", 0);
ArmPlugin::pass::ConvertShuffleChannels::ConvertShuffleChannels() {
    auto shuffle = ngraph::pattern::wrap_type<opset::ShuffleChannels>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto shuffle = std::dynamic_pointer_cast<opset::ShuffleChannels>(m.get_match_root());

        if (!shuffle) {
            return false;
        }

        int axis = shuffle->get_axis();
        if (axis < 0) {
            axis += shuffle->get_input_shape(0).size();
        }

        if (axis == 1) {
            return false;
        }

        std::vector<int64_t> input_order(shuffle->get_input_shape(0).size());
        std::iota(input_order.begin(), input_order.end(), 0);
        std::swap(input_order[axis], input_order[1]);

        auto order = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);

        size_t group = shuffle->get_group();
        auto tr_forward = std::make_shared<opset::Transpose>(shuffle->input_value(0), order);
        auto shuffle_new = std::make_shared<opset::ShuffleChannels>(tr_forward, 1, group);
        auto tr_backward = std::make_shared<opset::Transpose>(shuffle_new, order);

        tr_backward->set_friendly_name(shuffle->get_friendly_name());
        ngraph::copy_runtime_info(shuffle, {tr_forward, shuffle_new, tr_backward});
        ngraph::replace_node(shuffle, tr_backward);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(shuffle, "ConvertShuffleChannels");
    register_matcher(m, callback);
}
