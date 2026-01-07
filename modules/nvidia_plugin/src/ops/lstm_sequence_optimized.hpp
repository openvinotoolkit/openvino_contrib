// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lstm_sequence_base.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Implements `ov::nvidia_gpu::nodes::LSTMSequenceOptimized` using cuDNN API.
 *
 * OpenVINO and cuDNN are using different layouts for input/output tensor
 * shapes. In general, it takes 5 additional transpose operations to
 * adapt OpenVINO layouts to cuDNN and back. So, we are using graph
 * transformations to fuse Transpose operators before and after rnn sequence
 * in order to minimize the number of Transposes.
 *
 * cuDNN API supports the following layouts (assuming projection size is
 * equal to hidden size):
 *  - Batch Major Unpacked layout:
 *      cx, hx, cy, hy: [num_directions, batch_size, hidden_size]
 *      x:              [batch_size, seq_length, input_size]
 *      y:              [batch_size, seq_length, num_directions, hidden_size]
 *  - Time Major Unpacked layout (same as Legacy Packed layout but not packed):
 *      cx, hx, cy, hy: [num_directions, batch_size, hidden_size]
 *      x:              [seq_length, batch_size, input_size]
 *      y:              [seq_length, batch_size, num_directions, hidden_size]
 */
class LSTMSequenceOptimizedOp : public LSTMSequenceOpBase {
public:
    using NodeOp = ov::nvidia_gpu::nodes::LSTMSequenceOptimized;
    LSTMSequenceOptimizedOp(const CreationContext& context,
                            const NodeOp& node,
                            IndexCollection&& inputIds,
                            IndexCollection&& outputIds);

private:
    static LSTMSequenceOpBase::Config config(const NodeOp& node);
    void validateBatchMajorArgShapes(const NodeOp& node);
    void setupBatchMajorLayoutAdapters();
    void validateSequenceMajorArgShapes(const NodeOp& node);
    void setupSequenceMajorLayoutAdapters();
};

}  // namespace nvidia_gpu
}  // namespace ov
