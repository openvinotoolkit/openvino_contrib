// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gsl/span>
#include <ngraph/op/gru_cell.hpp>
#include <ngraph/op/lstm_cell.hpp>

namespace CUDAPlugin::RNN::Details {

/**
 * @brief Defines tensor indices for `ngraph::op::v4::LSTMCell` node.
 */
struct LSTMCellArgIndices {
    static constexpr size_t x = 0;
    static constexpr size_t hidden_input = 1;
    static constexpr size_t cell_input = 2;
    static constexpr size_t weights = 3;
    static constexpr size_t recurrence_weights = 4;
    static constexpr size_t biases = 5;
    static constexpr size_t hidden_output = 0;
    static constexpr size_t cell_output = 1;
};

/**
 * @brief Unified LSTM Cell parameters as they are consumed by different
 * implementations.
 *
 * This class extracts and validates required parameter values from ngraph operation;
 */
struct LSTMCellParams {
    LSTMCellParams(const ngraph::Node& node);
    LSTMCellParams(const ngraph::op::v4::LSTMCell& cell);

    static constexpr int lin_layer_count = 4;

    std::size_t hidden_size_;
    std::vector<std::string> activations_;
    std::vector<float> activations_alpha_;
    std::vector<float> activations_beta_;
    float clip_;

    size_t input_size_;
    size_t batch_size_;
    ngraph::element::Type element_type_;
    gsl::span<const uint8_t> w_host_buffers_;
    gsl::span<const uint8_t> r_host_buffers_;
    gsl::span<const uint8_t> b_host_buffers_;
};

/**
 * @brief Defines tensor indices for `ngraph::op::v4::LSTMCell` node.
 */
struct GRUCellArgIndices {
    static constexpr size_t x = 0;
    static constexpr size_t hidden_input = 1;
    static constexpr size_t weights = 2;
    static constexpr size_t recurrence_weights = 3;
    static constexpr size_t biases = 4;
    static constexpr size_t hidden_output = 0;
};

/**
 * @brief Unified GRU Cell parameters as they are consumed by different
 * implementations.
 *
 * This class extracts and validates required parameter values from ngraph operation;
 */
struct GRUCellParams {
    GRUCellParams(const ngraph::Node& node);
    GRUCellParams(const ngraph::op::v3::GRUCell& cell);

    static constexpr int lin_layer_count = 3;

    std::size_t hidden_size_;
    std::vector<std::string> activations_;
    std::vector<float> activations_alpha_;
    std::vector<float> activations_beta_;
    float clip_;
    bool linear_before_reset_;

    size_t input_size_;
    size_t batch_size_;
    ngraph::element::Type element_type_;
    gsl::span<const uint8_t> w_host_buffers_;
    gsl::span<const uint8_t> r_host_buffers_;
    gsl::span<const uint8_t> b_host_buffers_;
};

}  // namespace CUDAPlugin::RNN::Details
