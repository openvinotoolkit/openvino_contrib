// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_rnn_cell.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

enum RNNInput {InputData, HiddenState, Weights, RecurrenceWeights, Bias};

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertRNNCell, "ConvertRNNCell", 0);
ArmPlugin::pass::ConvertRNNCell::ConvertRNNCell() {
    auto rnn_cell = ngraph::pattern::wrap_type<opset::RNNCell>();
    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        auto rnn_cell = std::dynamic_pointer_cast<opset::RNNCell>(m.get_match_root());
        if (!rnn_cell || transformation_callback(rnn_cell)) {
            return false;
        }

        auto input_data = rnn_cell->input_value(RNNInput::InputData).get_node_shared_ptr();
        auto hidden_state = rnn_cell->input_value(RNNInput::HiddenState).get_node_shared_ptr();
        auto weights = rnn_cell->input_value(RNNInput::Weights).get_node_shared_ptr();
        auto recurrence_weights = rnn_cell->input_value(RNNInput::RecurrenceWeights).get_node_shared_ptr();
        auto bias = rnn_cell->input_value(RNNInput::Bias).get_node_shared_ptr();

        auto clip = rnn_cell->get_clip();

        if (clip == 0.f) {
            return false;
        }
        ov::NodeVector new_ops;
        input_data = std::make_shared<opset::Clamp>(input_data, -clip, clip);
        input_data->set_friendly_name(rnn_cell->get_friendly_name() + "/clip");
        new_ops.push_back(input_data);

        auto new_rnn_cell = std::make_shared<opset::RNNCell>(input_data,
                                                             hidden_state,
                                                             weights,
                                                             recurrence_weights,
                                                             bias,
                                                             rnn_cell->get_hidden_size(),
                                                             rnn_cell->get_activations(),
                                                             rnn_cell->get_activations_alpha(),
                                                             rnn_cell->get_activations_beta());
        new_ops.push_back(new_rnn_cell);
        new_rnn_cell->set_friendly_name(new_rnn_cell->get_friendly_name() + "/new");
        ngraph::copy_runtime_info(new_rnn_cell, new_ops);
        ngraph::replace_node(rnn_cell, new_rnn_cell);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(rnn_cell, "ConvertRNNCell");
    register_matcher(m, callback);
}