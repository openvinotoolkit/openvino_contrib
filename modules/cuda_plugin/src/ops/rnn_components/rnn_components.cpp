// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn_components.hpp"

#include <error.hpp>
#include <gsl/gsl_assert>
#include <openvino/op/constant.hpp>
#include <typeinfo>

namespace CUDAPlugin::RNN::Details {

namespace {

template <typename T>
const T& toRNNCell(const ov::Node& node) {
    static_assert(std::is_base_of_v<ov::op::util::RNNCellBase, T>, "T node should have base ov::op::util::RNNCellBase");
    try {
        return dynamic_cast<const T&>(node);
    } catch (const std::bad_cast&) {
        throwIEException("Couldn't convert ov::Node node to the derived T of base ov::op::util::RNNCellBase");
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
        Expects(cell.get_input_partial_shape(i).rank().is_static());
    }

    Expects(input_count == 6);
    Expects(cell.get_output_size() == 2);

    const auto& x_shape = cell.get_input_shape(LSTMCellArgIndices::x);
    Expects(x_shape.size() == 2);
    input_size_ = x_shape[1];
    batch_size_ = x_shape[0];

    const auto& hi_shape = cell.get_input_shape(LSTMCellArgIndices::hidden_input);
    Expects(hi_shape.size() == 2);
    Expects(hi_shape[0] == batch_size_);
    Expects(hi_shape[1] == hidden_size_);

    const auto& ci_shape = cell.get_input_shape(LSTMCellArgIndices::cell_input);
    Expects(ci_shape.size() == 2);
    Expects(ci_shape[0] == batch_size_);
    Expects(ci_shape[1] == hidden_size_);

    const auto& w_shape = cell.get_input_shape(LSTMCellArgIndices::weights);
    Expects(w_shape.size() == 2);
    Expects(w_shape[0] == lin_layer_count * hidden_size_);
    Expects(w_shape[1] == input_size_);

    const auto& r_shape = cell.get_input_shape(LSTMCellArgIndices::recurrence_weights);
    Expects(r_shape.size() == 2);
    Expects(r_shape[0] == lin_layer_count * hidden_size_);
    Expects(r_shape[1] == hidden_size_);

    element_type_ = cell.get_input_element_type(LSTMCellArgIndices::x);
    Expects(cell.get_input_element_type(LSTMCellArgIndices::hidden_input) == element_type_ &&
            cell.get_input_element_type(LSTMCellArgIndices::cell_input) == element_type_ &&
            cell.get_input_element_type(LSTMCellArgIndices::weights) == element_type_ &&
            cell.get_input_element_type(LSTMCellArgIndices::recurrence_weights) == element_type_ &&
            cell.get_output_element_type(LSTMCellArgIndices::hidden_output) == element_type_ &&
            cell.get_output_element_type(LSTMCellArgIndices::cell_output) == element_type_);

    const auto b_shape = cell.get_input_shape(LSTMCellArgIndices::biases);
    Expects(b_shape.size() == 1);
    Expects(b_shape[0] == lin_layer_count * hidden_size_);
    Expects(cell.get_input_element_type(LSTMCellArgIndices::biases) == element_type_);

    const auto& ho_shape = cell.get_output_shape(LSTMCellArgIndices::hidden_output);
    Expects(ho_shape.size() == 2);
    Expects(ho_shape[0] == batch_size_);
    Expects(ho_shape[1] == hidden_size_);

    const auto& co_shape = cell.get_output_shape(LSTMCellArgIndices::cell_output);
    Expects(co_shape.size() == 2);
    Expects(co_shape[0] == batch_size_);
    Expects(co_shape[1] == hidden_size_);

    const auto element_type_size = element_type_.size();

    const auto w_constant =
        dynamic_cast<ov::op::v0::Constant*>(cell.get_input_node_ptr(LSTMCellArgIndices::weights));
    const auto r_constant =
        dynamic_cast<ov::op::v0::Constant*>(cell.get_input_node_ptr(LSTMCellArgIndices::recurrence_weights));
    const auto b_constant =
        dynamic_cast<ov::op::v0::Constant*>(cell.get_input_node_ptr(LSTMCellArgIndices::biases));
    Expects(w_constant && r_constant && b_constant);

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
        Expects(cell.get_input_partial_shape(i).rank().is_static());
    }

    Expects(input_count == 4);
    Expects(cell.get_output_size() == 1);

    const auto& x_shape = cell.get_input_shape(GRUCellArgIndices::x);
    Expects(x_shape.size() == 2);
    input_size_ = x_shape[1];
    batch_size_ = x_shape[0];

    const auto& hi_shape = cell.get_input_shape(GRUCellArgIndices::hidden_input);
    Expects(hi_shape.size() == 2);
    Expects(hi_shape[0] == batch_size_);
    Expects(hi_shape[1] == hidden_size_);

    const auto& w_shape = cell.get_input_shape(GRUCellArgIndices::weights);
    Expects(w_shape.size() == 2);
    Expects(w_shape[0] == lin_layer_count * hidden_size_);
    Expects(w_shape[1] == input_size_);

    const auto& r_shape = cell.get_input_shape(GRUCellArgIndices::recurrence_weights);
    Expects(r_shape.size() == 2);
    Expects(r_shape[0] == lin_layer_count * hidden_size_);
    Expects(r_shape[1] == hidden_size_);

    element_type_ = cell.get_input_element_type(GRUCellArgIndices::x);
    Expects(cell.get_input_element_type(GRUCellArgIndices::hidden_input) == element_type_ &&
            cell.get_input_element_type(GRUCellArgIndices::weights) == element_type_ &&
            cell.get_input_element_type(GRUCellArgIndices::recurrence_weights) == element_type_ &&
            cell.get_output_element_type(GRUCellArgIndices::hidden_output) == element_type_);

    const auto b_shape = cell.get_input_shape(GRUCellArgIndices::biases);
    Expects(b_shape.size() == 1);
    if (cell.get_linear_before_reset()) {
        Expects(b_shape[0] == (lin_layer_count + 1) * hidden_size_);
    } else {
        Expects(b_shape[0] == lin_layer_count * hidden_size_);
    }
    Expects(cell.get_input_element_type(GRUCellArgIndices::biases) == element_type_);

    const auto& ho_shape = cell.get_output_shape(GRUCellArgIndices::hidden_output);
    Expects(ho_shape.size() == 2);
    Expects(ho_shape[0] == batch_size_);
    Expects(ho_shape[1] == hidden_size_);

    const auto element_type_size = element_type_.size();

    const auto w_constant =
        dynamic_cast<ov::op::v0::Constant*>(cell.get_input_node_ptr(GRUCellArgIndices::weights));
    const auto r_constant =
        dynamic_cast<ov::op::v0::Constant*>(cell.get_input_node_ptr(GRUCellArgIndices::recurrence_weights));
    const auto b_constant = dynamic_cast<ov::op::v0::Constant*>(cell.get_input_node_ptr(GRUCellArgIndices::biases));
    Expects(w_constant && r_constant && b_constant);

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
}  // namespace CUDAPlugin::RNN::Details
