// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_rnn_cell.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

enum RNNInput {InputData, HiddenState, Weights, RecurrenceWeights, Bias};

OPENVINO_OP(ArmPlugin::pass::ConvertRNNCell, "ConvertRNNCell");
ArmPlugin::pass::ConvertRNNCell::ConvertRNNCell() {
    auto rnn_cell = ngraph::pattern::wrap_type<opset::RNNCell>();
    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        auto rnn_cell = std::dynamic_pointer_cast<opset::RNNCell>(m.get_match_root());
        if (!rnn_cell || transformation_callback(rnn_cell)) {
            return false;
        }

        auto name_activation = rnn_cell->get_activations()[0];
        if ((name_activation == "tanh" || name_activation == "sigmoid") && (rnn_cell->get_clip() == 0.f)) {
            return false;
        }

        if (name_activation == "relu" && rnn_cell->get_clip() != 0.f) {
            return false;
        }

        auto X = rnn_cell->input_value(InputData);
        auto H_t = rnn_cell->input_value(HiddenState);
        auto W = rnn_cell->input_value(Weights);
        auto R = rnn_cell->input_value(RecurrenceWeights);
        auto bias = rnn_cell->input_value(Bias);

        // Xt*(W^T)
        auto Xt_W = std::make_shared<opset::MatMul>(X, W, false, true);
        // Ht-1*(R^T)
        auto Ht_R = std::make_shared<opset::MatMul>(H_t, R, false, true);
        // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
        auto add = std::make_shared<opset::Add>(Ht_R, bias);
        auto i_t = std::make_shared<opset::Add>(Xt_W, add);

        // f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
        auto clip = rnn_cell->get_clip();
        std::shared_ptr<ngraph::Node> clamp = i_t;
        if (clip > 0.f) {
            clamp = std::make_shared<opset::Clamp>(i_t, -clip, clip);
            ngraph::copy_runtime_info(rnn_cell, clamp);
        }
        auto out = ov::op::util::activation(rnn_cell->get_activations()[0], clamp);
        out->set_friendly_name(rnn_cell->get_friendly_name());
        ngraph::copy_runtime_info(rnn_cell, {Xt_W, Ht_R, add, i_t, out});
        ngraph::replace_node(rnn_cell, out);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(rnn_cell, "ConvertRNNCell");
    register_matcher(m, callback);
}
