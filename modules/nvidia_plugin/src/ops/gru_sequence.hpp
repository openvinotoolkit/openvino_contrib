// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <openvino/op/gru_sequence.hpp>
#include <ops/components/workbuffer_desc.hpp>

#include "rnn_components/gru_sequence_components.hpp"
#include "rnn_components/gru_sequence_cudnn_components.hpp"
#include "rnn_components/rnn_sequence_components.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Implements `ov::op::v5::GRUSequence` using cuDNN API
 */
class GRUSequenceOp : public OperationCuDnn {
public:
    using NodeOp = ov::op::v5::GRUSequence;
    using Config = RNN::Details::GRUSequenceDescriptorsCuDnn::Config;
    GRUSequenceOp(const CreationContext& context,
                  const NodeOp& node,
                  IndexCollection&& inputIds,
                  IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;

private:
    static Config config();
    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override;
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    const RNN::Details::GRUSequenceParamsCuDnn params_;
    RNN::Details::GRUSequenceDescriptorsCuDnn descs_;

    std::vector<WorkbufferRequest::size_in_bytes_t> immut_sizes_;
    std::vector<WorkbufferRequest::size_in_bytes_t> mut_sizes_;

    WorkbufferDesc ib_seq_lengths_;
    WorkbufferDesc ib_weight_space_;
    WorkbufferDesc mb_work_space_;

    CudaGraphCompatibility graph_compatibility_;
};

}  // namespace nvidia_gpu
}  // namespace ov
