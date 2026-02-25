// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn_components.hpp"

#include <error.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>
#include <typeinfo>

namespace ov::nvidia_gpu::RNN::Details {

namespace {

template <typename T>
const T& toRNNCell(const ov::Node& node) {
    static_assert(std::is_base_of_v<ov::op::util::RNNCellBase, T>, "T node should have base ov::op::util::RNNCellBase");
    try {
        return dynamic_cast<const T&>(node);
    } catch (const std::bad_cast&) {
        throw_ov_exception("Couldn't convert ov::Node node to the derived T of base ov::op::util::RNNCellBase");
    }
}

}  // namespace

LSTMCellParams::LSTMCellParams(const ov::Node& node) : LSTMCellParams(toRNNCell<ov::op::v4::LSTMCell>(node)) {}

LSTMCellParams::LSTMCellParams(const ov::op::v4::LSTMCell& cell)
    : hidden_size_{cell.get_hidden_size()},
      activations_{cell.get_activations()},
      activations_alpha_{cell.get_activations_alpha()},
      activations_beta_{cell.get_activations_beta()},
      clip_{cell.get_clip()} {
    const auto input_count = cell.get_input_size();
    for (int i = 0; i < input_count; ++i) {
        OPENVINO_ASSERT(cell.get_input_partial_shape(i).rank().is_static());
    }

    OPENVINO_ASSERT(input_count == 6);
    OPENVINO_ASSERT(cell.get_output_size() == 2);

    const auto& x_shape = cell.get_input_shape(LSTMCellArgIndices::x);
    OPENVINO_ASSERT(x_shape.size() == 2);
    input_size_ = x_shape[1];
    batch_size_ = x_shape[0];

    const auto& hi_shape = cell.get_input_shape(LSTMCellArgIndices::hidden_input);
    OPENVINO_ASSERT(hi_shape.size() == 2);
    OPENVINO_ASSERT(hi_shape[0] == batch_size_);
    OPENVINO_ASSERT(hi_shape[1] == hidden_size_);

    const auto& ci_shape = cell.get_input_shape(LSTMCellArgIndices::cell_input);
    OPENVINO_ASSERT(ci_shape.size() == 2);
    OPENVINO_ASSERT(ci_shape[0] == batch_size_);
    OPENVINO_ASSERT(ci_shape[1] == hidden_size_);

    const auto& w_shape = cell.get_input_shape(LSTMCellArgIndices::weights);
    OPENVINO_ASSERT(w_shape.size() == 2);
    OPENVINO_ASSERT(w_shape[0] == lin_layer_count * hidden_size_);
    OPENVINO_ASSERT(w_shape[1] == input_size_);

    const auto& r_shape = cell.get_input_shape(LSTMCellArgIndices::recurrence_weights);
    OPENVINO_ASSERT(r_shape.size() == 2);
    OPENVINO_ASSERT(r_shape[0] == lin_layer_count * hidden_size_);
    OPENVINO_ASSERT(r_shape[1] == hidden_size_);

    element_type_ = cell.get_input_element_type(LSTMCellArgIndices::x);
    OPENVINO_ASSERT(cell.get_input_element_type(LSTMCellArgIndices::hidden_input) == element_type_ &&
            cell.get_input_element_type(LSTMCellArgIndices::cell_input) == element_type_ &&
            cell.get_input_element_type(LSTMCellArgIndices::weights) == element_type_ &&
            cell.get_input_element_type(LSTMCellArgIndices::recurrence_weights) == element_type_ &&
            cell.get_output_element_type(LSTMCellArgIndices::hidden_output) == element_type_ &&
            cell.get_output_element_type(LSTMCellArgIndices::cell_output) == element_type_);

    const auto b_shape = cell.get_input_shape(LSTMCellArgIndices::biases);
    OPENVINO_ASSERT(b_shape.size() == 1);
    OPENVINO_ASSERT(b_shape[0] == lin_layer_count * hidden_size_);
    OPENVINO_ASSERT(cell.get_input_element_type(LSTMCellArgIndices::biases) == element_type_);

    const auto& ho_shape = cell.get_output_shape(LSTMCellArgIndices::hidden_output);
    OPENVINO_ASSERT(ho_shape.size() == 2);
    OPENVINO_ASSERT(ho_shape[0] == batch_size_);
    OPENVINO_ASSERT(ho_shape[1] == hidden_size_);

    const auto& co_shape = cell.get_output_shape(LSTMCellArgIndices::cell_output);
    OPENVINO_ASSERT(co_shape.size() == 2);
    OPENVINO_ASSERT(co_shape[0] == batch_size_);
    OPENVINO_ASSERT(co_shape[1] == hidden_size_);

    const auto element_type_size = element_type_.size();

    const auto w_constant = dynamic_cast<ov::op::v0::Constant*>(cell.get_input_node_ptr(LSTMCellArgIndices::weights));
    const auto r_constant =
        dynamic_cast<ov::op::v0::Constant*>(cell.get_input_node_ptr(LSTMCellArgIndices::recurrence_weights));
    const auto b_constant = dynamic_cast<ov::op::v0::Constant*>(cell.get_input_node_ptr(LSTMCellArgIndices::biases));
    OPENVINO_ASSERT(w_constant && r_constant && b_constant);

    const auto w_data_host = w_constant->get_data_ptr<const uint8_t>();
    const auto r_data_host = r_constant->get_data_ptr<const uint8_t>();
    const auto b_data_host = b_constant->get_data_ptr<const uint8_t>();

    const std::size_t w_size_bytes = ov::shape_size(w_shape) * element_type_size;
    const std::size_t r_size_bytes = ov::shape_size(r_shape) * element_type_size;
    const std::size_t b_size_bytes = ov::shape_size(b_shape) * element_type_size;

    w_host_buffers_ = {w_data_host, w_size_bytes};
    r_host_buffers_ = {r_data_host, r_size_bytes};
    b_host_buffers_ = {b_data_host, b_size_bytes};
}

GRUCellParams::GRUCellParams(const ov::Node& node) : GRUCellParams(toRNNCell<ov::op::v3::GRUCell>(node)) {}

GRUCellParams::GRUCellParams(const ov::op::v3::GRUCell& cell)
    : hidden_size_{cell.get_hidden_size()},
      activations_{cell.get_activations()},
      activations_alpha_{cell.get_activations_alpha()},
      activations_beta_{cell.get_activations_beta()},
      clip_{cell.get_clip()},
      linear_before_reset_{cell.get_linear_before_reset()} {
    const auto input_count = cell.get_input_size() - 1;
    for (int i = 0; i < input_count; ++i) {
        OPENVINO_ASSERT(cell.get_input_partial_shape(i).rank().is_static());
    }

    OPENVINO_ASSERT(input_count == 4);
    OPENVINO_ASSERT(cell.get_output_size() == 1);

    const auto& x_shape = cell.get_input_shape(GRUCellArgIndices::x);
    OPENVINO_ASSERT(x_shape.size() == 2);
    input_size_ = x_shape[1];
    batch_size_ = x_shape[0];

    const auto& hi_shape = cell.get_input_shape(GRUCellArgIndices::hidden_input);
    OPENVINO_ASSERT(hi_shape.size() == 2);
    OPENVINO_ASSERT(hi_shape[0] == batch_size_);
    OPENVINO_ASSERT(hi_shape[1] == hidden_size_);

    const auto& w_shape = cell.get_input_shape(GRUCellArgIndices::weights);
    OPENVINO_ASSERT(w_shape.size() == 2);
    OPENVINO_ASSERT(w_shape[0] == lin_layer_count * hidden_size_);
    OPENVINO_ASSERT(w_shape[1] == input_size_);

    const auto& r_shape = cell.get_input_shape(GRUCellArgIndices::recurrence_weights);
    OPENVINO_ASSERT(r_shape.size() == 2);
    OPENVINO_ASSERT(r_shape[0] == lin_layer_count * hidden_size_);
    OPENVINO_ASSERT(r_shape[1] == hidden_size_);

    element_type_ = cell.get_input_element_type(GRUCellArgIndices::x);
    OPENVINO_ASSERT(cell.get_input_element_type(GRUCellArgIndices::hidden_input) == element_type_ &&
            cell.get_input_element_type(GRUCellArgIndices::weights) == element_type_ &&
            cell.get_input_element_type(GRUCellArgIndices::recurrence_weights) == element_type_ &&
            cell.get_output_element_type(GRUCellArgIndices::hidden_output) == element_type_);

    const auto b_shape = cell.get_input_shape(GRUCellArgIndices::biases);
    OPENVINO_ASSERT(b_shape.size() == 1);
    if (cell.get_linear_before_reset()) {
        OPENVINO_ASSERT(b_shape[0] == (lin_layer_count + 1) * hidden_size_);
    } else {
        OPENVINO_ASSERT(b_shape[0] == lin_layer_count * hidden_size_);
    }
    OPENVINO_ASSERT(cell.get_input_element_type(GRUCellArgIndices::biases) == element_type_);

    const auto& ho_shape = cell.get_output_shape(GRUCellArgIndices::hidden_output);
    OPENVINO_ASSERT(ho_shape.size() == 2);
    OPENVINO_ASSERT(ho_shape[0] == batch_size_);
    OPENVINO_ASSERT(ho_shape[1] == hidden_size_);

    const auto element_type_size = element_type_.size();

    const auto w_constant = dynamic_cast<ov::op::v0::Constant*>(cell.get_input_node_ptr(GRUCellArgIndices::weights));
    const auto r_constant =
        dynamic_cast<ov::op::v0::Constant*>(cell.get_input_node_ptr(GRUCellArgIndices::recurrence_weights));
    const auto b_constant = dynamic_cast<ov::op::v0::Constant*>(cell.get_input_node_ptr(GRUCellArgIndices::biases));
    OPENVINO_ASSERT(w_constant && r_constant && b_constant);

    const auto w_data_host = w_constant->get_data_ptr<const uint8_t>();
    const auto r_data_host = r_constant->get_data_ptr<const uint8_t>();
    const auto b_data_host = b_constant->get_data_ptr<const uint8_t>();

    const std::size_t w_size_bytes = ov::shape_size(w_shape) * element_type_size;
    const std::size_t r_size_bytes = ov::shape_size(r_shape) * element_type_size;
    const std::size_t b_size_bytes = ov::shape_size(b_shape) * element_type_size;

    w_host_buffers_ = {w_data_host, w_size_bytes};
    r_host_buffers_ = {r_data_host, r_size_bytes};
    b_host_buffers_ = {b_data_host, b_size_bytes};
}
}  // namespace ov::nvidia_gpu::RNN::Details
