// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gru_sequence_components.hpp"

#include <cuda_op_buffers_extractor.hpp>
#include <error.hpp>
#include <gsl/gsl_assert>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <typeinfo>

namespace CUDAPlugin::RNN::Details {

namespace {

gsl::span<const uint8_t> findInputConstantBuffer(const ov::Node& inNode, int inputIdx) {
    const ov::Node* node = inNode.get_input_node_ptr(inputIdx);
    const ov::op::v0::Constant* constant = dynamic_cast<const ov::op::v0::Constant*>(node);
    while (!constant) {
        Expects(OperationBuffersExtractor::isReshapeOnlyNode(*node));
        node = node->get_input_node_ptr(0);
        constant = dynamic_cast<const ov::op::v0::Constant*>(node);
    }
    Expects(constant);
    const size_t size_bytes =
        ov::shape_size(constant->get_output_shape(0)) * constant->get_output_element_type(0).size();
    return {constant->get_data_ptr<const uint8_t>(), size_bytes};
}

}  // namespace

GRUSequenceParams::GRUSequenceParams(const ov::op::v5::GRUSequence& node)
    : element_type_{node.get_input_element_type(GRUSequenceArgIndices::x)},
      direction_{node.get_direction()},
      activations_{node.get_activations()},
      clip_{node.get_clip()},
      hidden_size_{node.get_hidden_size()},
      linear_before_reset_{node.get_linear_before_reset()} {
    Expects(node.get_input_size() == 6);
    Expects(node.get_output_size() == 2);

    const auto input_count = node.get_input_size();
    for (size_t i = 0; i < input_count; ++i) {
        Expects(node.get_input_partial_shape(i).rank().is_static());
    }

    const auto& x_shape = node.get_input_shape(GRUSequenceArgIndices::x);
    Expects(x_shape.size() == 3);
    batch_size_ = x_shape[0];
    max_seq_length_ = x_shape[1];
    input_size_ = x_shape[2];

    Expects(node.get_input_element_type(GRUSequenceArgIndices::x) == element_type_ &&
            node.get_input_element_type(GRUSequenceArgIndices::hidden_input) == element_type_ &&
            node.get_input_element_type(GRUSequenceArgIndices::weights) == element_type_ &&
            node.get_input_element_type(GRUSequenceArgIndices::recurrence_weights) == element_type_ &&
            node.get_input_element_type(GRUSequenceArgIndices::biases) == element_type_ &&
            node.get_output_element_type(GRUSequenceArgIndices::y) == element_type_ &&
            node.get_output_element_type(GRUSequenceArgIndices::hidden_output) == element_type_);

    w_host_buffers_ = findInputConstantBuffer(node, GRUSequenceArgIndices::weights);
    r_host_buffers_ = findInputConstantBuffer(node, GRUSequenceArgIndices::recurrence_weights);
    b_host_buffers_ = findInputConstantBuffer(node, GRUSequenceArgIndices::biases);

    const size_t num_directions = (direction_ == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL) ? 2 : 1;

    const auto& w_shape = node.get_input_shape(GRUSequenceArgIndices::weights);
    Expects(w_shape.size() == 3);
    Expects(w_shape[0] == num_directions);
    Expects(w_shape[1] == lin_layer_count * hidden_size_);
    Expects(w_shape[2] == input_size_);

    const auto& r_shape = node.get_input_shape(GRUSequenceArgIndices::recurrence_weights);
    Expects(r_shape.size() == 3);
    Expects(r_shape[0] == num_directions);
    Expects(r_shape[1] == lin_layer_count * hidden_size_);
    Expects(r_shape[2] == hidden_size_);

    const auto& b_shape = node.get_input_shape(GRUSequenceArgIndices::biases);
    Expects(b_shape.size() == 2);
    Expects(b_shape[0] == num_directions);
    if (node.get_linear_before_reset()) {
        Expects(b_shape[1] == (lin_layer_count + 1) * hidden_size_);
    } else {
        Expects(b_shape[1] == lin_layer_count * hidden_size_);
    }

    const auto element_type_size = element_type_.size();
    Expects(w_host_buffers_.size_bytes() == ov::shape_size(w_shape) * element_type_size);
    Expects(r_host_buffers_.size_bytes() == ov::shape_size(r_shape) * element_type_size);
    Expects(b_host_buffers_.size_bytes() == ov::shape_size(b_shape) * element_type_size);
}

}  // namespace CUDAPlugin::RNN::Details
