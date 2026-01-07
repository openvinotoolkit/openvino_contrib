// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lstm_sequence_base.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Implements `ov::op::v5::LSTMSequence` using cuDNN API
 */
class LSTMSequenceOp : public LSTMSequenceOpBase {
public:
    using NodeOp = ov::op::v5::LSTMSequence;
    LSTMSequenceOp(const CreationContext& context,
                   const NodeOp& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds);

private:
    static LSTMSequenceOpBase::Config config();
    void setupLayoutAdapters();
};

}  // namespace nvidia_gpu
}  // namespace ov
