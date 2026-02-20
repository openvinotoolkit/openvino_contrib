// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_sequence_optimized.hpp"

#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <utility>
#include <vector>

namespace ov {
namespace nvidia_gpu {

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
            break;
        default:
            OPENVINO_ASSERT(false, "Node name: ", GetName());
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
            OPENVINO_ASSERT(false);
    };

    return config;
}

void LSTMSequenceOptimizedOp::validateBatchMajorArgShapes(const NodeOp& node) {
    using LSTMSequenceArgIndices = ov::nvidia_gpu::RNN::Details::LSTMSequenceArgIndices;
    const int64_t batch_size = params_.batch_size_;
    const int64_t num_directions = params_.numDirections();
    const int64_t hidden_size = params_.hidden_size_;
    const int64_t input_size = params_.input_size_;
    const int64_t max_seq_length = params_.max_seq_length_;

    const auto& x_shape = node.get_input_shape(LSTMSequenceArgIndices::x);
    OPENVINO_ASSERT(x_shape.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(x_shape[0] == batch_size, "Node name: ", GetName());
    OPENVINO_ASSERT(x_shape[1] == max_seq_length, "Node name: ", GetName());
    OPENVINO_ASSERT(x_shape[2] == input_size, "Node name: ", GetName());

    const auto& hx_shape = node.get_input_shape(LSTMSequenceArgIndices::hidden_input);
    OPENVINO_ASSERT(hx_shape.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(hx_shape[0] == batch_size, "Node name: ", GetName());
    OPENVINO_ASSERT(hx_shape[1] == num_directions, "Node name: ", GetName());
    OPENVINO_ASSERT(hx_shape[2] == hidden_size, "Node name: ", GetName());

    const auto& cx_shape = node.get_input_shape(LSTMSequenceArgIndices::cell_input);
    OPENVINO_ASSERT(cx_shape.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(cx_shape[0] == batch_size, "Node name: ", GetName());
    OPENVINO_ASSERT(cx_shape[1] == num_directions, "Node name: ", GetName());
    OPENVINO_ASSERT(cx_shape[2] == hidden_size, "Node name: ", GetName());

    const auto& y_shape = node.get_output_shape(LSTMSequenceArgIndices::y);
    OPENVINO_ASSERT(y_shape.size() == 4, "Node name: ", GetName());
    OPENVINO_ASSERT(y_shape[0] == batch_size, "Node name: ", GetName());
    OPENVINO_ASSERT(y_shape[1] == max_seq_length, "Node name: ", GetName());
    OPENVINO_ASSERT(y_shape[2] == num_directions, "Node name: ", GetName());
    OPENVINO_ASSERT(y_shape[3] == hidden_size, "Node name: ", GetName());

    const auto& hy_shape = node.get_output_shape(LSTMSequenceArgIndices::hidden_output);
    OPENVINO_ASSERT(hy_shape.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(hy_shape[0] == num_directions, "Node name: ", GetName());
    OPENVINO_ASSERT(hy_shape[1] == batch_size, "Node name: ", GetName());
    OPENVINO_ASSERT(hy_shape[2] == hidden_size, "Node name: ", GetName());

    const auto& cy_shape = node.get_output_shape(LSTMSequenceArgIndices::cell_output);
    OPENVINO_ASSERT(cy_shape.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(cy_shape[0] == num_directions, "Node name: ", GetName());
    OPENVINO_ASSERT(cy_shape[1] == batch_size, "Node name: ", GetName());
    OPENVINO_ASSERT(cy_shape[2] == hidden_size, "Node name: ", GetName());
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
    using LSTMSequenceArgIndices = ov::nvidia_gpu::RNN::Details::LSTMSequenceArgIndices;
    const int64_t batch_size = params_.batch_size_;
    const int64_t num_directions = params_.numDirections();
    const int64_t hidden_size = params_.hidden_size_;
    const int64_t input_size = params_.input_size_;
    const int64_t max_seq_length = params_.max_seq_length_;

    const auto& x_shape = node.get_input_shape(LSTMSequenceArgIndices::x);
    OPENVINO_ASSERT(x_shape.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(x_shape[0] == max_seq_length, "Node name: ", GetName());
    OPENVINO_ASSERT(x_shape[1] == batch_size, "Node name: ", GetName());
    OPENVINO_ASSERT(x_shape[2] == input_size, "Node name: ", GetName());

    const auto& hx_shape = node.get_input_shape(LSTMSequenceArgIndices::hidden_input);
    OPENVINO_ASSERT(hx_shape.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(hx_shape[0] == batch_size, "Node name: ", GetName());
    OPENVINO_ASSERT(hx_shape[1] == num_directions, "Node name: ", GetName());
    OPENVINO_ASSERT(hx_shape[2] == hidden_size, "Node name: ", GetName());

    const auto& cx_shape = node.get_input_shape(LSTMSequenceArgIndices::cell_input);
    OPENVINO_ASSERT(cx_shape.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(cx_shape[0] == batch_size, "Node name: ", GetName());
    OPENVINO_ASSERT(cx_shape[1] == num_directions, "Node name: ", GetName());
    OPENVINO_ASSERT(cx_shape[2] == hidden_size, "Node name: ", GetName());

    const auto& y_shape = node.get_output_shape(LSTMSequenceArgIndices::y);
    OPENVINO_ASSERT(y_shape.size() == 4, "Node name: ", GetName());
    OPENVINO_ASSERT(y_shape[0] == max_seq_length, "Node name: ", GetName());
    OPENVINO_ASSERT(y_shape[1] == batch_size, "Node name: ", GetName());
    OPENVINO_ASSERT(y_shape[2] == num_directions, "Node name: ", GetName());
    OPENVINO_ASSERT(y_shape[3] == hidden_size, "Node name: ", GetName());

    const auto& hy_shape = node.get_output_shape(LSTMSequenceArgIndices::hidden_output);
    OPENVINO_ASSERT(hy_shape.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(hy_shape[0] == num_directions, "Node name: ", GetName());
    OPENVINO_ASSERT(hy_shape[1] == batch_size, "Node name: ", GetName());
    OPENVINO_ASSERT(hy_shape[2] == hidden_size, "Node name: ", GetName());

    const auto& cy_shape = node.get_output_shape(LSTMSequenceArgIndices::cell_output);
    OPENVINO_ASSERT(cy_shape.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(cy_shape[0] == num_directions, "Node name: ", GetName());
    OPENVINO_ASSERT(cy_shape[1] == batch_size, "Node name: ", GetName());
    OPENVINO_ASSERT(cy_shape[2] == hidden_size, "Node name: ", GetName());
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

}  // namespace nvidia_gpu
}  // namespace ov
