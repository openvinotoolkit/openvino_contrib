// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_sequence_optimized.hpp"

#include <cuda_operation_registry.hpp>
#include <gsl/gsl_assert>
#include <utility>
#include <vector>

namespace CUDAPlugin {

LSTMSequenceOptimizedOp::LSTMSequenceOptimizedOp(const CreationContext& context,
                                                 const NodeOp& node,
                                                 IndexCollection&& inputIds,
                                                 IndexCollection&& outputIds)
    : LSTMSequenceOpBase(
          context, LSTMSequenceParams{node}, config(node), node, std::move(inputIds), std::move(outputIds)) {
    switch (node.get_major_format()) {
        case NodeOp::BatchMajor:
            /*
                For this optimization operator shapes are as following:
                    in [X]          - [batch_size, seq_length, input_size]
                    cell/hidden in  - [batch_size, num_directions, hidden_size]
                    out [Y]         - [batch_size, seq_length, num_directions, hidden_size]
                    cell/hidden out - [num_directions, batch_size, hidden_size]
            */
            validateBatchMajorArgShapes(node);
            setupBatchMajorLayoutAdapters();
            break;
        case NodeOp::SequenceMajor:
            /*
                For this optimization operator shapes are as following:
                    in [X]          - [seq_length, batch_size, input_size]
                    cell/hidden in  - [batch_size, num_directions, hidden_size]
                    out [Y]         - [seq_length, batch_size, num_directions, hidden_size]
                    cell/hidden out - [num_directions, batch_size, hidden_size]
            */
            validateSequenceMajorArgShapes(node);
            setupSequenceMajorLayoutAdapters();
            throwIEException(
                "'CUDAPlugin::nodes::LSTMSequenceOptimized::SequenceMajor': This mode has never been used with real "
                "model.");
            break;
        default:
            Expects(false);
    };
    calcAdapterWorkbuffers();
}

LSTMSequenceOpBase::Config LSTMSequenceOptimizedOp::config(const NodeOp& node) {
    LSTMSequenceOpBase::Config config{};
    switch (node.get_major_format()) {
        case NodeOp::BatchMajor:
            config.rnn_data_layout = CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
            break;
        case NodeOp::SequenceMajor:
            config.rnn_data_layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
            break;
        default:
            Expects(false);
    };

    return config;
}

void LSTMSequenceOptimizedOp::validateBatchMajorArgShapes(const NodeOp& node) {
    using LSTMSequenceArgIndices = CUDAPlugin::RNN::Details::LSTMSequenceArgIndices;
    const int64_t batch_size = params_.batch_size_;
    const int64_t num_directions = params_.numDirections();
    const int64_t hidden_size = params_.hidden_size_;
    const int64_t input_size = params_.input_size_;
    const int64_t max_seq_length = params_.max_seq_length_;

    const auto& x_shape = node.get_input_shape(LSTMSequenceArgIndices::x);
    Expects(x_shape.size() == 3);
    Expects(x_shape[0] == batch_size);
    Expects(x_shape[1] == max_seq_length);
    Expects(x_shape[2] == input_size);

    const auto& hx_shape = node.get_input_shape(LSTMSequenceArgIndices::hidden_input);
    Expects(hx_shape.size() == 3);
    Expects(hx_shape[0] == batch_size);
    Expects(hx_shape[1] == num_directions);
    Expects(hx_shape[2] == hidden_size);

    const auto& cx_shape = node.get_input_shape(LSTMSequenceArgIndices::cell_input);
    Expects(cx_shape.size() == 3);
    Expects(cx_shape[0] == batch_size);
    Expects(cx_shape[1] == num_directions);
    Expects(cx_shape[2] == hidden_size);

    const auto& y_shape = node.get_output_shape(LSTMSequenceArgIndices::y);
    Expects(y_shape.size() == 4);
    Expects(y_shape[0] == batch_size);
    Expects(y_shape[1] == max_seq_length);
    Expects(y_shape[2] == num_directions);
    Expects(y_shape[3] == hidden_size);

    const auto& hy_shape = node.get_output_shape(LSTMSequenceArgIndices::hidden_output);
    Expects(hy_shape.size() == 3);
    Expects(hy_shape[0] == num_directions);
    Expects(hy_shape[1] == batch_size);
    Expects(hy_shape[2] == hidden_size);

    const auto& cy_shape = node.get_output_shape(LSTMSequenceArgIndices::cell_output);
    Expects(cy_shape.size() == 3);
    Expects(cy_shape[0] == num_directions);
    Expects(cy_shape[1] == batch_size);
    Expects(cy_shape[2] == hidden_size);
}

void LSTMSequenceOptimizedOp::setupBatchMajorLayoutAdapters() {
    using InputAdapter = RNN::Details::TransposeInputTensorAdapter;
    using OutputAdapter = RNN::Details::TransposeOutputTensorAdapter;

    const int64_t batch_size = params_.batch_size_;
    const int64_t num_directions = params_.numDirections();
    const int64_t hidden_size = params_.hidden_size_;

    const std::vector<int64_t> state_shape_op = {batch_size, num_directions, hidden_size};
    const std::vector<int64_t> state_shape_cudnn = {num_directions, batch_size, hidden_size};
    const bool transposeStateInputs = (batch_size > 1) && (num_directions > 1);
    if (transposeStateInputs) {
        hx_adapter = std::make_unique<InputAdapter>(params_.element_type_cuda_,
                                                    params_.element_size_,
                                                    state_shape_op,
                                                    state_shape_cudnn,
                                                    std::vector<int>{1, 0, 2});
        cx_adapter = std::make_unique<InputAdapter>(params_.element_type_cuda_,
                                                    params_.element_size_,
                                                    state_shape_op,
                                                    state_shape_cudnn,
                                                    std::vector<int>{1, 0, 2});
    }
}

void LSTMSequenceOptimizedOp::validateSequenceMajorArgShapes(const NodeOp& node) {
    using LSTMSequenceArgIndices = CUDAPlugin::RNN::Details::LSTMSequenceArgIndices;
    const int64_t batch_size = params_.batch_size_;
    const int64_t num_directions = params_.numDirections();
    const int64_t hidden_size = params_.hidden_size_;
    const int64_t input_size = params_.input_size_;
    const int64_t max_seq_length = params_.max_seq_length_;

    const auto& x_shape = node.get_input_shape(LSTMSequenceArgIndices::x);
    Expects(x_shape.size() == 3);
    Expects(x_shape[0] == max_seq_length);
    Expects(x_shape[1] == batch_size);
    Expects(x_shape[2] == input_size);

    const auto& hx_shape = node.get_input_shape(LSTMSequenceArgIndices::hidden_input);
    Expects(hx_shape.size() == 3);
    Expects(hx_shape[0] == batch_size);
    Expects(hx_shape[1] == num_directions);
    Expects(hx_shape[2] == hidden_size);

    const auto& cx_shape = node.get_input_shape(LSTMSequenceArgIndices::cell_input);
    Expects(cx_shape.size() == 3);
    Expects(cx_shape[0] == batch_size);
    Expects(cx_shape[1] == num_directions);
    Expects(cx_shape[2] == hidden_size);

    const auto& y_shape = node.get_output_shape(LSTMSequenceArgIndices::y);
    Expects(y_shape.size() == 4);
    Expects(y_shape[0] == max_seq_length);
    Expects(y_shape[1] == batch_size);
    Expects(y_shape[2] == num_directions);
    Expects(y_shape[3] == hidden_size);

    const auto& hy_shape = node.get_output_shape(LSTMSequenceArgIndices::hidden_output);
    Expects(hy_shape.size() == 3);
    Expects(hy_shape[0] == num_directions);
    Expects(hy_shape[1] == batch_size);
    Expects(hy_shape[2] == hidden_size);

    const auto& cy_shape = node.get_output_shape(LSTMSequenceArgIndices::cell_output);
    Expects(cy_shape.size() == 3);
    Expects(cy_shape[0] == num_directions);
    Expects(cy_shape[1] == batch_size);
    Expects(cy_shape[2] == hidden_size);
}

void LSTMSequenceOptimizedOp::setupSequenceMajorLayoutAdapters() {
    using InputAdapter = RNN::Details::TransposeInputTensorAdapter;
    using OutputAdapter = RNN::Details::TransposeOutputTensorAdapter;

    const int64_t batch_size = params_.batch_size_;
    const int64_t num_directions = params_.numDirections();
    const int64_t hidden_size = params_.hidden_size_;

    const std::vector<int64_t> state_shape_op = {batch_size, num_directions, hidden_size};
    const std::vector<int64_t> state_shape_cudnn = {num_directions, batch_size, hidden_size};
    const bool transposeStateInputs = (batch_size > 1) && (num_directions > 1);
    if (transposeStateInputs) {
        hx_adapter = std::make_unique<InputAdapter>(params_.element_type_cuda_,
                                                    params_.element_size_,
                                                    state_shape_op,
                                                    state_shape_cudnn,
                                                    std::vector<int>{1, 0, 2});
        cx_adapter = std::make_unique<InputAdapter>(params_.element_type_cuda_,
                                                    params_.element_size_,
                                                    state_shape_op,
                                                    state_shape_cudnn,
                                                    std::vector<int>{1, 0, 2});
    }
}

OPERATION_REGISTER(LSTMSequenceOptimizedOp, LSTMSequenceOptimized);

}  // namespace CUDAPlugin
