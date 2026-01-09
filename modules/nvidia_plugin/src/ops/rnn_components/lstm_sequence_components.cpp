// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_sequence_components.hpp"

#include <cuda/constant_factory.hpp>
#include <cuda_op_buffers_extractor.hpp>
#include <error.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <typeinfo>

namespace ov::nvidia_gpu::RNN::Details {

namespace {

gsl::span<const uint8_t> findInputConstantBuffer(const ov::Node& inNode, int inputIdx) {
    const ov::Node* node = inNode.get_input_node_ptr(inputIdx);
    const ov::op::v0::Constant* constant = dynamic_cast<const ov::op::v0::Constant*>(node);
    while (!constant) {
        OPENVINO_ASSERT(OperationBuffersExtractor::isReshapeOnlyNode(*node));
        node = node->get_input_node_ptr(0);
        constant = dynamic_cast<const ov::op::v0::Constant*>(node);
    }
    OPENVINO_ASSERT(constant);
    const size_t size_bytes =
        ov::shape_size(constant->get_output_shape(0)) * constant->get_output_element_type(0).size();
    return {constant->get_data_ptr<const uint8_t>(), size_bytes};
}

}  // namespace

LSTMSequenceParams::LSTMSequenceParams(const ov::op::v5::LSTMSequence& node)
    : element_type_{node.get_input_element_type(LSTMSequenceArgIndices::x)},
      direction_{node.get_direction()},
      activations_{node.get_activations()},
      activations_alpha_{node.get_activations_alpha()},
      activations_beta_{node.get_activations_beta()},
      clip_{node.get_clip()},
      hidden_size_{node.get_hidden_size()} {
    OPENVINO_ASSERT(node.get_input_size() == 7);
    OPENVINO_ASSERT(node.get_output_size() == 3);

    const auto& x_shape = node.get_input_shape(LSTMSequenceArgIndices::x);
    OPENVINO_ASSERT(x_shape.size() == 3);
    batch_size_ = x_shape[0];
    max_seq_length_ = x_shape[1];
    input_size_ = x_shape[2];

    w_host_buffers_ = findInputConstantBuffer(node, LSTMSequenceArgIndices::weights);
    r_host_buffers_ = findInputConstantBuffer(node, LSTMSequenceArgIndices::recurrence_weights);
    b_host_buffers_ = findInputConstantBuffer(node, LSTMSequenceArgIndices::biases);

    validate(node);
}

LSTMSequenceParams::LSTMSequenceParams(const ov::nvidia_gpu::nodes::LSTMSequenceOptimized& node)
    : element_type_{node.get_input_element_type(LSTMSequenceArgIndices::x)},
      direction_{node.get_direction()},
      activations_{node.get_activations()},
      activations_alpha_{node.get_activations_alpha()},
      activations_beta_{node.get_activations_beta()},
      clip_{node.get_clip()},
      hidden_size_{node.get_hidden_size()} {
    OPENVINO_ASSERT(node.get_input_size() == 7);
    OPENVINO_ASSERT(node.get_output_size() == 3);

    const auto& x_shape = node.get_input_shape(LSTMSequenceArgIndices::x);
    OPENVINO_ASSERT(x_shape.size() == 3);
    using LSTMSequenceOptimized = ov::nvidia_gpu::nodes::LSTMSequenceOptimized;
    switch (node.get_major_format()) {
        case LSTMSequenceOptimized::BatchMajor:
            batch_size_ = x_shape[0];
            max_seq_length_ = x_shape[1];
            input_size_ = x_shape[2];
            break;
        case LSTMSequenceOptimized::SequenceMajor:
            max_seq_length_ = x_shape[0];
            batch_size_ = x_shape[1];
            input_size_ = x_shape[2];
            break;
        default:
            OPENVINO_ASSERT(false);
    }

    w_host_buffers_ = findInputConstantBuffer(node, LSTMSequenceArgIndices::weights);
    r_host_buffers_ = findInputConstantBuffer(node, LSTMSequenceArgIndices::recurrence_weights);
    b_host_buffers_ = findInputConstantBuffer(node, LSTMSequenceArgIndices::biases);

    validate(node);
}

void LSTMSequenceParams::validate(const ov::op::util::RNNCellBase& node) {
    const auto& sl_shape = node.get_input_shape(LSTMSequenceArgIndices::sequence_lengths);
    OPENVINO_ASSERT(sl_shape.size() == 1);
    OPENVINO_ASSERT(sl_shape[0] == batch_size_);

    OPENVINO_ASSERT(node.get_input_element_type(LSTMSequenceArgIndices::x) == element_type_ &&
            node.get_input_element_type(LSTMSequenceArgIndices::hidden_input) == element_type_ &&
            node.get_input_element_type(LSTMSequenceArgIndices::cell_input) == element_type_ &&
            node.get_input_element_type(LSTMSequenceArgIndices::weights) == element_type_ &&
            node.get_input_element_type(LSTMSequenceArgIndices::recurrence_weights) == element_type_ &&
            node.get_input_element_type(LSTMSequenceArgIndices::biases) == element_type_ &&
            node.get_output_element_type(LSTMSequenceArgIndices::y) == element_type_ &&
            node.get_output_element_type(LSTMSequenceArgIndices::hidden_output) == element_type_ &&
            node.get_output_element_type(LSTMSequenceArgIndices::cell_output) == element_type_);

    const size_t num_directions = (direction_ == direction::BIDIRECTIONAL) ? 2 : 1;

    const auto& w_shape = node.get_input_shape(LSTMSequenceArgIndices::weights);
    OPENVINO_ASSERT(w_shape.size() == 3);
    OPENVINO_ASSERT(w_shape[0] == num_directions);
    OPENVINO_ASSERT(w_shape[1] == lin_layer_count * hidden_size_);
    OPENVINO_ASSERT(w_shape[2] == input_size_);

    const auto& r_shape = node.get_input_shape(LSTMSequenceArgIndices::recurrence_weights);
    OPENVINO_ASSERT(r_shape.size() == 3);
    OPENVINO_ASSERT(r_shape[0] == num_directions);
    OPENVINO_ASSERT(r_shape[1] == lin_layer_count * hidden_size_);
    OPENVINO_ASSERT(r_shape[2] == hidden_size_);

    const auto& b_shape = node.get_input_shape(LSTMSequenceArgIndices::biases);
    OPENVINO_ASSERT(b_shape.size() == 2);
    OPENVINO_ASSERT(b_shape[0] == num_directions);
    OPENVINO_ASSERT(b_shape[1] == lin_layer_count * hidden_size_);

    const auto element_type_size = element_type_.size();
    OPENVINO_ASSERT(w_host_buffers_.size_bytes() == ov::shape_size(w_shape) * element_type_size);
    OPENVINO_ASSERT(r_host_buffers_.size_bytes() == ov::shape_size(r_shape) * element_type_size);
    OPENVINO_ASSERT(b_host_buffers_.size_bytes() == ov::shape_size(b_shape) * element_type_size);
}

}  // namespace ov::nvidia_gpu::RNN::Details
