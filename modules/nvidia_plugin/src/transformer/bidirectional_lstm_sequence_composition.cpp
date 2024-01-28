// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bidirectional_lstm_sequence_composition.hpp"

#include "cuda_op_buffers_extractor.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/op_types.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformer/nodes/concat_optimized.hpp"
#include "transformer/nodes/lstm_sequence_optimized.hpp"


using namespace ov::pass;
using namespace ov::pass::pattern;

namespace ov::nvidia_gpu::pass {
namespace {

std::vector<int64_t> get_transpose_order(const std::shared_ptr<ov::op::v1::Transpose>& transpose) {
    if (transpose) {
        auto transpose_const = ov::as_type_ptr<op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
        if (transpose_const) {
            return transpose_const->cast_vector<int64_t>();
        }
    }
    return std::vector<int64_t>{};
}

bool is_y0_batch_reshape(const ov::Shape& input_shape, const ov::Shape& output_shape) {
    if (input_shape.size() != 4 || output_shape.size() != 4) {
        return false;
    }
    if (input_shape[0] == output_shape[0] && input_shape[1] == output_shape[2] &&
        input_shape[2] == output_shape[1] && input_shape[3] == output_shape[3]) {
        return true;
    }
    return false;
}

bool is_y0_batch_transpose(const std::shared_ptr<Node>& node) {
    if (node) {
        if (auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(node)) {
            if (get_transpose_order(transpose) == std::vector<int64_t>{0, 2, 1, 3}) {
                return true;
            }
        }
        if (auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(node)) {
            if (reshape->is_dynamic()) {
                return false;
            }
            if (is_y0_batch_reshape(reshape->input(0).get_shape(),
                                    reshape->output(0).get_shape())) {
                return true;
            }
        }
    }
    return false;
}

bool is_y0_sequence_reshape(const ov::Shape& input_shape, const ov::Shape& output_shape) {
    if (input_shape.size() != 4 || output_shape.size() != 4) {
        return false;
    }
    if (input_shape[0] == output_shape[1] && input_shape[1] == output_shape[2] &&
        input_shape[2] == output_shape[0] && input_shape[3] == output_shape[3]) {
        return true;
    }
    return false;
}

bool is_y0_sequence_transpose(const std::shared_ptr<Node>& node) {
    if (node) {
        if (auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(node)) {
            if (get_transpose_order(transpose) == std::vector<int64_t>{2, 0, 1, 3}) {
                return true;
            }
        }
        if (auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(node)) {
            if (reshape->is_dynamic()) {
                return false;
            }
            if (is_y0_sequence_reshape(reshape->input(0).get_shape(),
                                       reshape->output(0).get_shape())) {
                return true;
            }
        }
    }
    return false;
}

bool is_h0_c0_reshape(const ov::Shape& input_shape, const ov::Shape& output_shape) {
    if (input_shape.size() != 3 || output_shape.size() != 3) {
        return false;
    }
    if (input_shape[0] == output_shape[1] && input_shape[1] == output_shape[0] &&
        input_shape[2] == output_shape[2]) {
        return true;
    }
    return false;
}

bool is_h0_c0_transpose(const std::shared_ptr<Node>& node) {
    if (node) {
        if (auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(node)) {
            if (get_transpose_order(transpose) == std::vector<int64_t>{1, 0, 2}) {
                return true;
            }
        }
        if (auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(node)) {
            if (reshape->is_dynamic()) {
                return false;
            }
            if (is_h0_c0_reshape(reshape->input(0).get_shape(),
                                 reshape->output(0).get_shape())) {
                return true;
            }
        }
    }
    return false;
}

bool is_required_transpose_or_reshape(const std::shared_ptr<ov::Node>& node) {
    return (is_y0_batch_transpose(node) ||
            is_y0_sequence_transpose(node) ||
            is_h0_c0_transpose(node) ||
            ov::as_type_ptr<ov::op::v1::Reshape>(node));
}

std::shared_ptr<ov::Node> find_concat(const Output<Node>& node, std::shared_ptr<Node>& transpose) {
    for (const auto& in : node.get_target_inputs()) {
        auto current_node = in.get_node()->shared_from_this();
        if (!transpose && is_required_transpose_or_reshape(current_node)) {
            transpose = current_node;
        }
        if (ov::as_type_ptr<ov::op::v0::Concat>(current_node) ||
            ov::as_type_ptr<ov::nvidia_gpu::nodes::ConcatOptimized>(current_node)) {
            return current_node;
        }
        for (const auto& out : current_node->outputs()) {
            auto found_node = find_concat(out, transpose);
            if (found_node) {
                return found_node;
            }
        }
    }
    return nullptr;
}

void get_last_transpose_or_reshape(const std::shared_ptr<ov::Node>& node, std::shared_ptr<ov::Node>& found_node) {
    auto check_node = [&] (const std::shared_ptr<ov::Node>& node) -> bool {
        if (is_required_transpose_or_reshape(node)) {
            found_node = node;
            return true;
        }
        return false;
    };
    if (check_node(node)) {
        auto consumers = node->get_output_target_inputs(0);
        if (1 == consumers.size()) {
            get_last_transpose_or_reshape(consumers.begin()->get_node()->shared_from_this(), found_node);
        }
    }
}

bool is_direction(const std::shared_ptr<ov::Node>& node,
                  const ov::op::RecurrentSequenceDirection& direction) {
    if (auto lstm_sequence = ov::as_type_ptr<ov::op::v5::LSTMSequence>(node)) {
        if (lstm_sequence->get_direction() == direction) {
            return true;
        }
    }
    return false;
}

bool is_bidirectional(const std::shared_ptr<ov::Node>& node) {
    return is_direction(node, ov::op::RecurrentSequenceDirection::BIDIRECTIONAL);
}

bool is_forward(const std::shared_ptr<ov::Node>& node) {
    return is_direction(node, ov::op::RecurrentSequenceDirection::FORWARD);
}

bool is_reverse(const std::shared_ptr<ov::Node>& node) {
    return is_direction(node, ov::op::RecurrentSequenceDirection::REVERSE);
}

std::vector<int64_t> gen_y_transpose_order(const std::shared_ptr<Node>& node) {
    if (is_y0_batch_transpose(node)) {
        return std::vector<int64_t>{0, 2, 1, 3};
    }
    if (is_y0_sequence_transpose(node)) {
        return std::vector<int64_t>{2, 0, 1, 3};
    }
    return std::vector<int64_t>{};
}

void output_replacer(const std::shared_ptr<ov::Node>& replacement,
                     const ov::Output<ov::Node>& new_out) {
    if (!replacement) {
        return;
    }
    for (auto& out : replacement->outputs()) {
        for (const auto& in : out.get_target_inputs()) {
            const auto& in_node = in.get_node();
            in.replace_source_output(new_out);
        }
    }
};
}  // namespace

bool bidirectional_lstm_sequence_composition(
    const std::shared_ptr<ov::Node>& x,
    const std::shared_ptr<ov::op::v5::LSTMSequence>& lstm_sequence_forward,
    const std::shared_ptr<ov::op::v5::LSTMSequence>& lstm_sequence_reverse) {

    if (lstm_sequence_forward->get_hidden_size() != lstm_sequence_reverse->get_hidden_size() ||
        lstm_sequence_forward->get_activations_alpha() != lstm_sequence_reverse->get_activations_alpha() ||
        lstm_sequence_forward->get_activations_beta() != lstm_sequence_reverse->get_activations_beta() ||
        lstm_sequence_forward->get_activations() != lstm_sequence_reverse->get_activations() ||
        lstm_sequence_forward->get_clip() != lstm_sequence_reverse->get_clip()) {
        return false;
    }

    std::shared_ptr<Node> y_transpose = nullptr;
    std::shared_ptr<Node> ho_transpose = nullptr;
    std::shared_ptr<Node> co_transpose = nullptr;
    auto y_replacement = find_concat(lstm_sequence_forward->output(0), y_transpose);
    auto ho_replacement = find_concat(lstm_sequence_forward->output(1), ho_transpose);
    auto co_replacement = find_concat(lstm_sequence_forward->output(2), co_transpose);
    if (!y_replacement || !ho_replacement || !co_replacement) {
        return false;
    }
    if (!(!y_transpose || is_y0_batch_transpose(y_transpose) || is_y0_sequence_transpose(y_transpose)) ||
        !(!ho_transpose || is_h0_c0_transpose(ho_transpose)) ||
        !(!co_transpose || is_h0_c0_transpose(co_transpose))) {
        return false;
    }
    auto concat_inputs = [&lstm_sequence_forward, &lstm_sequence_reverse](size_t index, size_t axis) {
        return std::make_shared<ov::op::v0::Concat>(ov::OutputVector{
            lstm_sequence_forward->input_value(index),
            lstm_sequence_reverse->input_value(index)}, axis);
    };
    auto lstm_sequence_bidirectional =
        std::make_shared<ov::op::v5::LSTMSequence>(x,
            concat_inputs(1, 1),
            concat_inputs(2, 1),
            lstm_sequence_forward->input_value(3),
            concat_inputs(4, 0),
            concat_inputs(5, 0),
            concat_inputs(6, 0),
            lstm_sequence_forward->get_hidden_size(),
            ov::op::RecurrentSequenceDirection::BIDIRECTIONAL,
            lstm_sequence_forward->get_activations_alpha(),
            lstm_sequence_forward->get_activations_beta(),
            lstm_sequence_forward->get_activations(),
            lstm_sequence_forward->get_clip());
    auto y = lstm_sequence_bidirectional->output(0);
    auto ho = lstm_sequence_bidirectional->output(1);
    auto co = lstm_sequence_bidirectional->output(2);
    ov::copy_runtime_info({lstm_sequence_forward, lstm_sequence_reverse}, lstm_sequence_bidirectional);

    std::string original_names_mapping;
    if (y_transpose) {
        auto transpose_y_const =
            std::make_shared<ov::op::v0::Constant>(y_transpose->input_value(1).get_element_type(),
            ov::Shape{4}, gen_y_transpose_order(y_transpose));
        auto transpose_ho_co_const =
            std::make_shared<ov::op::v0::Constant>(ho_transpose ? ho_transpose->input_value(1).get_element_type() : ov::element::i32,
            ov::Shape{3}, std::vector<int64_t>{1, 0, 2});
        auto transpose_y = std::make_shared<ov::op::v1::Transpose>(y, transpose_y_const);
        auto transpose_ho = std::make_shared<ov::op::v1::Transpose>(ho, transpose_ho_co_const);
        auto transpose_co = std::make_shared<ov::op::v1::Transpose>(co, transpose_ho_co_const);
        transpose_y->set_friendly_name(y_replacement->get_friendly_name());
        transpose_ho->set_friendly_name(ho_replacement->get_friendly_name());
        transpose_co->set_friendly_name(co_replacement->get_friendly_name());
        ov::copy_runtime_info(y_replacement, transpose_y);
        ov::copy_runtime_info(ho_replacement, transpose_ho);
        ov::copy_runtime_info(co_replacement, transpose_co);
        output_replacer(y_replacement, transpose_y->output(0));
        output_replacer(ho_replacement, transpose_ho->output(0));
        output_replacer(co_replacement, transpose_co->output(0));
    } else {
        output_replacer(y_replacement, y);
        output_replacer(ho_replacement, ho);
        output_replacer(co_replacement, co);
    }
    return true;
}

bool bidirectional_lstm_sequence_cudnn_optimized(const std::shared_ptr<ov::op::v5::LSTMSequence>& lstm_sequence_bidirectional) {
    // Original OpenVINO:
    //   in              - [batch_size, seq_length, input_size]
    //   out             - [batch_size, num_directions, seq_length, hidden_size]
    //   cell/hidden in  - [batch_size, num_directions, hidden_size]
    //   cell/hidden out - [batch_size, num_directions, hidden_size]
    //
    // cuDNN Optimized 0:
    //   in              - [batch_size, seq_length, input_size]
    //   out             - [batch_size, seq_length, num_directions, hidden_size]
    //   cell/hidden in  - [batch_size, num_directions, hidden_size]
    //   cell/hidden out - [num_directions, batch_size, hidden_size]
    //
    // cuDNN Optimized 1:
    //   in              - [seq_length, batch_size, input_size]
    //   out             - [seq_length, batch_size, num_directions, hidden_size]
    //   cell/hidden in  - [batch_size, num_directions, hidden_size]
    //   cell/hidden out - [num_directions, batch_size, hidden_size]

    using MajorFormat = ov::nvidia_gpu::nodes::LSTMSequenceOptimized::MajorFormat;

    auto get_replacement = [&lstm_sequence_bidirectional](size_t index, std::shared_ptr<ov::Node>& replacement) {
        auto consumers = lstm_sequence_bidirectional->get_output_target_inputs(index);
        if (1 == consumers.size()) {
            get_last_transpose_or_reshape(consumers.begin()->get_node()->shared_from_this(), replacement);
        }
    };
    std::vector<std::shared_ptr<Node>> nodes_to_be_replaced = { lstm_sequence_bidirectional };
    std::shared_ptr<Node> y_replacement = nullptr;
    std::shared_ptr<Node> ho_replacement = nullptr;
    std::shared_ptr<Node> co_replacement = nullptr;
    get_replacement(0, y_replacement);
    get_replacement(1, ho_replacement);
    get_replacement(2, co_replacement);
    if (!y_replacement) {
        return false;
    }
    nodes_to_be_replaced.push_back(y_replacement);
    const auto y_shape = lstm_sequence_bidirectional->output(0).get_shape();
    const auto y_replacement_shape = y_replacement->output(0).get_shape();
    if ((y_replacement_shape.size() < 3) || (y_replacement_shape.size() > 4)) {
        return false;
    }
    auto check_ho_co_replacement = [&](size_t index, const std::shared_ptr<Node>& replacement) {
        if (replacement) {
            if (!is_h0_c0_reshape(lstm_sequence_bidirectional->output(index).get_shape(),
                                  replacement->output(0).get_shape())) {
                return false;
            }
            nodes_to_be_replaced.push_back(replacement);
        } else if (lstm_sequence_bidirectional->get_output_target_inputs(index).size() > 0) {
            return false;
        }
        return true;
    };
    if (!check_ho_co_replacement(1, ho_replacement) || !check_ho_co_replacement(2, co_replacement)) {
        return false;
    }
    std::optional<MajorFormat> major_format;
    std::shared_ptr<ov::Node> x_transpose = nullptr;
    if ((y_replacement_shape[0] == y_shape[0]) && (y_replacement_shape[1] == y_shape[2]) &&
        ((y_replacement_shape.size() == 3) || is_y0_batch_reshape(y_shape, y_replacement_shape))) {
        if (y_replacement->get_input_source_output(0).get_node()->shared_from_this() == lstm_sequence_bidirectional) {
            if (is_y0_batch_transpose(y_replacement)) {
                major_format = MajorFormat::BatchMajor;
            }
        } else {
            major_format = MajorFormat::BatchMajor;
        }
    }
    if (!major_format) {
        if ((y_replacement_shape[0] == y_shape[2]) && (y_replacement_shape[1] == y_shape[0]) &&
            ((y_replacement_shape.size() == 3) || is_y0_sequence_reshape(y_shape, y_replacement_shape))) {
            x_transpose = lstm_sequence_bidirectional->get_input_source_output(0).get_node()->shared_from_this();
            if (is_h0_c0_transpose(x_transpose)) {
                major_format = MajorFormat::SequenceMajor;
                nodes_to_be_replaced.push_back(x_transpose);
            }
        }
    }
    if (!major_format) {
        return false;
    }
    auto new_lstm_sequence_bidirectional = std::make_shared<ov::nvidia_gpu::nodes::LSTMSequenceOptimized>(
        x_transpose ? x_transpose->get_input_source_output(0) : lstm_sequence_bidirectional->get_input_source_output(0),
        lstm_sequence_bidirectional->get_input_source_output(1),
        lstm_sequence_bidirectional->get_input_source_output(2),
        lstm_sequence_bidirectional->get_input_source_output(3),
        lstm_sequence_bidirectional->get_input_source_output(4),
        lstm_sequence_bidirectional->get_input_source_output(5),
        lstm_sequence_bidirectional->get_input_source_output(6),
        lstm_sequence_bidirectional->get_hidden_size(),
        ov::op::RecurrentSequenceDirection::BIDIRECTIONAL,
        major_format.value(),
        lstm_sequence_bidirectional->get_activations_alpha(),
        lstm_sequence_bidirectional->get_activations_beta(),
        lstm_sequence_bidirectional->get_activations(),
        lstm_sequence_bidirectional->get_clip());

    new_lstm_sequence_bidirectional->set_friendly_name(lstm_sequence_bidirectional->get_friendly_name());
    ov::copy_runtime_info(nodes_to_be_replaced, new_lstm_sequence_bidirectional);

    if (y_replacement_shape.size() == 3) {
        std::shared_ptr<ov::op::v1::Reshape> reshape;
        const auto& out = new_lstm_sequence_bidirectional->output(0);
        const auto& out_shape = out.get_shape();
        std::vector<int32_t> reshape_pattern_values = {static_cast<int32_t>(out_shape[0]),
                                                       static_cast<int32_t>(out_shape[1]),
                                                       static_cast<int32_t>(out_shape[2] * out_shape[3])};
        auto reshape_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, reshape_pattern_values);
        reshape = std::make_shared<ov::op::v1::Reshape>(out, reshape_pattern, true);
        reshape->set_friendly_name(y_replacement->get_friendly_name());
        ov::copy_runtime_info(y_replacement, reshape);
        output_replacer(y_replacement, reshape->output(0));
    } else {
        output_replacer(y_replacement, new_lstm_sequence_bidirectional->output(0));
    }
    output_replacer(ho_replacement, new_lstm_sequence_bidirectional->output(1));
    output_replacer(co_replacement, new_lstm_sequence_bidirectional->output(2));

    return true;
}

bool Convert2LSTMSequenceToBidirectionalLSTMSequence::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(Convert2LSTMSequenceToBidirectionalLSTMSequence);

    bool was_updated = false;
    for (const auto& op : f->get_ordered_ops()) {
        for (auto& output : op->outputs()) {
            if (output.get_target_inputs().size() == 2) {
                std::shared_ptr<ov::op::v5::LSTMSequence> lstm_sequence_forward = nullptr;
                std::shared_ptr<ov::op::v5::LSTMSequence> lstm_sequence_reverse = nullptr;
                for (auto& consumer : output.get_target_inputs()) {
                    auto node = consumer.get_node()->shared_from_this();
                    if (is_forward(node)) {
                        lstm_sequence_forward = ov::as_type_ptr<ov::op::v5::LSTMSequence>(node);
                    } else if (is_reverse(node)) {
                        lstm_sequence_reverse = ov::as_type_ptr<ov::op::v5::LSTMSequence>(node);
                    }
                }
                if (lstm_sequence_forward && lstm_sequence_reverse) {
                    was_updated |= bidirectional_lstm_sequence_composition(op, lstm_sequence_forward, lstm_sequence_reverse);
                }
            }
        }
    }
    return was_updated;
}

bool ConvertBidirectionalLSTMSequenceToBidirectionalLSTMSequenceOptimized::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(ConvertBidirectionalLSTMSequenceToBidirectionalLSTMSequenceOptimized);

    bool was_updated = false;
    for (const auto& op : f->get_ordered_ops()) {
        if (is_bidirectional(op)) {
            was_updated |= bidirectional_lstm_sequence_cudnn_optimized(ov::as_type_ptr<ov::op::v5::LSTMSequence>(op));
        }
    }
    return was_updated;
}

bool BidirectionalSequenceComposition::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(BidirectionalSequenceComposition)
    Manager manager;

    manager.register_pass<Convert2LSTMSequenceToBidirectionalLSTMSequence>();
    manager.register_pass<ov::pass::NopElimination>();
    manager.register_pass<ConvertBidirectionalLSTMSequenceToBidirectionalLSTMSequenceOptimized>();

    manager.run_passes(f);

    return false;
}

}  // namespace ov::nvidia_gpu::pass
