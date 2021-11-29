// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_sequence.hpp"

#include <cuda_operation_registry.hpp>
#include <gsl/gsl_assert>
#include <utility>
#include <vector>

#include "lstm_sequence_components.hpp"

namespace CUDAPlugin {

LSTMSequenceOp::LSTMSequenceOp(const CreationContext& context,
                               const NodeOp& node,
                               IndexCollection&& inputIds,
                               IndexCollection&& outputIds)
    : LSTMSequenceOpBase(context, LSTMSequenceParams{node}, config(), node, std::move(inputIds), std::move(outputIds)) {
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
    Expects(y_shape[1] == num_directions);
    Expects(y_shape[2] == max_seq_length);
    Expects(y_shape[3] == hidden_size);

    const auto& hy_shape = node.get_output_shape(LSTMSequenceArgIndices::hidden_output);
    Expects(hy_shape.size() == 3);
    Expects(hy_shape[0] == batch_size);
    Expects(hy_shape[1] == num_directions);
    Expects(hy_shape[2] == hidden_size);

    const auto& cy_shape = node.get_output_shape(LSTMSequenceArgIndices::cell_output);
    Expects(cy_shape.size() == 3);
    Expects(cy_shape[0] == batch_size);
    Expects(cy_shape[1] == num_directions);
    Expects(cy_shape[2] == hidden_size);

    setupLayoutAdapters();
    calcAdapterWorkbuffers();
}

LSTMSequenceOpBase::Config LSTMSequenceOp::config() {
    LSTMSequenceOpBase::Config config{};
    config.rnn_data_layout = CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
    return config;
}

void LSTMSequenceOp::setupLayoutAdapters() {
    using InputAdapter = RNN::Details::TransposeInputTensorAdapter;
    using OutputAdapter = RNN::Details::TransposeOutputTensorAdapter;

    const int64_t batch_size = params_.batch_size_;
    const int64_t num_directions = params_.numDirections();
    const int64_t hidden_size = params_.hidden_size_;
    const int64_t max_seq_length = params_.max_seq_length_;

    // Determine whether it's necessary to transpose input state tensors to make OpenVINO state inputs binary compatible
    // with cuDNN API.
    const std::vector<int64_t> state_shape_openvino = {batch_size, num_directions, hidden_size};
    const std::vector<int64_t> state_shape_cudnn = {num_directions, batch_size, hidden_size};
    const bool transposeStateInputs = (batch_size > 1) && (num_directions > 1);
    if (transposeStateInputs) {
        hx_adapter = std::make_unique<InputAdapter>(params_.element_type_cuda_,
                                                    params_.element_size_,
                                                    state_shape_openvino,
                                                    state_shape_cudnn,
                                                    std::vector<int>{1, 0, 2});
        cx_adapter = std::make_unique<InputAdapter>(params_.element_type_cuda_,
                                                    params_.element_size_,
                                                    state_shape_openvino,
                                                    state_shape_cudnn,
                                                    std::vector<int>{1, 0, 2});
    }

    // Determine whether it's necessary to transpose output state tensors to make cuDNN API outputs binary compatible
    // with OpenVINO.
    const bool transposeStateOutputs = (batch_size > 1) && (num_directions > 1);
    if (transposeStateOutputs) {
        hy_adapter = std::make_unique<OutputAdapter>(params_.element_type_cuda_,
                                                     params_.element_size_,
                                                     state_shape_cudnn,
                                                     state_shape_openvino,
                                                     std::vector<int>{1, 0, 2});
        cy_adapter = std::make_unique<OutputAdapter>(params_.element_type_cuda_,
                                                     params_.element_size_,
                                                     state_shape_cudnn,
                                                     state_shape_openvino,
                                                     std::vector<int>{1, 0, 2});
    }

    // Determine whether it's necessary to transpose output Y tensor to make cuDNN API output binary compatible with
    // OpenVINO.
    const std::vector<int64_t> y_shape_openvino = {batch_size, num_directions, max_seq_length, hidden_size};
    const std::vector<int64_t> y_shape_cudnn = {batch_size, max_seq_length, num_directions, hidden_size};
    const bool transposeY = (num_directions > 1) && (max_seq_length > 1);
    if (transposeY) {
        y_adapter = std::make_unique<OutputAdapter>(params_.element_type_cuda_,
                                                    params_.element_size_,
                                                    y_shape_cudnn,
                                                    y_shape_openvino,
                                                    std::vector<int>{0, 2, 1, 3});
    }
}

OPERATION_REGISTER(LSTMSequenceOp, LSTMSequence);

}  // namespace CUDAPlugin
