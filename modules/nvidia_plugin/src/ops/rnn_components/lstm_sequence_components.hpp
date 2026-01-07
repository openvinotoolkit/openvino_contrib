// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <gsl/span>
#include <openvino/op/lstm_sequence.hpp>
#include <transformer/nodes/lstm_sequence_optimized.hpp>

namespace ov::nvidia_gpu::RNN::Details {

/**
 * @brief Defines tensor indices for `ov::op::v5::LSTMSequence` node.
 */
struct LSTMSequenceArgIndices {
    static constexpr size_t x = 0;
    static constexpr size_t hidden_input = 1;
    static constexpr size_t cell_input = 2;
    static constexpr size_t sequence_lengths = 3;
    static constexpr size_t weights = 4;
    static constexpr size_t recurrence_weights = 5;
    static constexpr size_t biases = 6;
    static constexpr size_t y = 0;
    static constexpr size_t hidden_output = 1;
    static constexpr size_t cell_output = 2;
};

/**
 * @brief Extracted and validated parameters from ngraph operation.
 */
struct LSTMSequenceParams {
    LSTMSequenceParams(const ov::op::v5::LSTMSequence& node);
    LSTMSequenceParams(const ov::nvidia_gpu::nodes::LSTMSequenceOptimized& node);

    static constexpr int lin_layer_count = 4;

    ov::element::Type element_type_;
    using direction = ov::op::v5::LSTMSequence::direction;
    direction direction_;
    std::vector<std::string> activations_;
    std::vector<float> activations_alpha_;
    std::vector<float> activations_beta_;
    float clip_;

    gsl::span<const uint8_t> w_host_buffers_;
    gsl::span<const uint8_t> r_host_buffers_;
    gsl::span<const uint8_t> b_host_buffers_;

    size_t batch_size_;
    size_t max_seq_length_;
    size_t input_size_;
    size_t hidden_size_;

private:
    void validate(const ov::op::util::RNNCellBase& node);
};

}  // namespace ov::nvidia_gpu::RNN::Details
