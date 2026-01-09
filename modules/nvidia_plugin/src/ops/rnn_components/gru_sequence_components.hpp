// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <gsl/span>
#include <openvino/op/gru_sequence.hpp>

#include "gru_sequence_components.hpp"

namespace ov::nvidia_gpu::RNN::Details {

/**
 * @brief Defines tensor indices for `ov::op::v5::GRUSequence` node.
 */
struct GRUSequenceArgIndices {
    static constexpr size_t x = 0;
    static constexpr size_t hidden_input = 1;
    static constexpr size_t sequence_lengths = 2;
    static constexpr size_t weights = 3;
    static constexpr size_t recurrence_weights = 4;
    static constexpr size_t biases = 5;
    static constexpr size_t y = 0;
    static constexpr size_t hidden_output = 1;
};

/**
 * @brief Extracted and validated parameters from ngraph operation.
 */
struct GRUSequenceParams {
    GRUSequenceParams(const ov::op::v5::GRUSequence& node);

    static constexpr int lin_layer_count = 3;

    ov::element::Type element_type_;
    ov::op::RecurrentSequenceDirection direction_;
    std::vector<std::string> activations_;
    float clip_;
    bool linear_before_reset_;

    gsl::span<const uint8_t> w_host_buffers_;
    gsl::span<const uint8_t> r_host_buffers_;
    gsl::span<const uint8_t> b_host_buffers_;

    size_t batch_size_;
    size_t max_seq_length_;
    size_t input_size_;
    size_t hidden_size_;
};

}  // namespace ov::nvidia_gpu::RNN::Details
