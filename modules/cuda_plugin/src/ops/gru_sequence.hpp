// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <ngraph/node.hpp>

#include "gru_sequence_components.hpp"
#include "gru_sequence_cudnn_components.hpp"
#include "ngraph/op/gru_sequence.hpp"
#include "rnn_sequence_components.hpp"

namespace CUDAPlugin {

/**
 * @brief Implements `ngraph::op::v5::GRUSequence` using cuDNN API
 */
class GRUSequenceOp : public OperationCuDnn {
public:
    using NodeOp = ngraph::op::v5::GRUSequence;
    using Config = RNN::Details::GRUSequenceDescriptorsCuDnn::Config;
    GRUSequenceOp(const CreationContext& context,
                  const NodeOp& node,
                  IndexCollection&& inputIds,
                  IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override;

private:
    static Config config();
    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override;
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    const RNN::Details::GRUSequenceParamsCuDnn params_;
    RNN::Details::GRUSequenceDescriptorsCuDnn descs_;

    std::vector<WorkbufferRequest::size_in_bytes_t> immut_sizes_;
    std::vector<WorkbufferRequest::size_in_bytes_t> mut_sizes_;

    using WorkbufferDesc = RNN::Details::WorkbufferDesc;
    WorkbufferDesc ib_seq_lengths_;
    WorkbufferDesc ib_weight_space_;
    WorkbufferDesc mb_work_space_;
};

}  // namespace CUDAPlugin
