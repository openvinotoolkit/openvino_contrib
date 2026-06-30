// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <ops/components/workbuffer_desc.hpp>

#include "rnn_components/lstm_sequence_components.hpp"
#include "rnn_components/lstm_sequence_cudnn_components.hpp"
#include "rnn_components/rnn_sequence_components.hpp"

namespace ov {
namespace nvidia_gpu {

class LSTMSequenceOpBase : public OperationCuDnn {
public:
    using NodeOp = ov::op::util::RNNCellBase;
    using LSTMSequenceParams = RNN::Details::LSTMSequenceParams;
    using Config = RNN::Details::LSTMSequenceDescriptorsCuDnn::Config;
    LSTMSequenceOpBase(const CreationContext& context,
                       const LSTMSequenceParams& params,
                       const Config& config,
                       const NodeOp& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;
    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override;
    WorkbufferRequest GetWorkBufferRequest() const override;

protected:
    void calcAdapterWorkbuffers();

protected:
    const RNN::Details::LSTMSequenceParamsCuDnn params_;
    RNN::Details::LSTMSequenceDescriptorsCuDnn descs_;

    std::vector<WorkbufferRequest::size_in_bytes_t> immut_sizes_;
    std::vector<WorkbufferRequest::size_in_bytes_t> mut_sizes_;

    WorkbufferDesc ib_seq_lengths_;
    WorkbufferDesc ib_weight_space_;
    WorkbufferDesc mb_work_space_;

    using InputTensorAdapterPtr = std::unique_ptr<RNN::Details::TransposeInputTensorAdapter>;
    InputTensorAdapterPtr x_adapter;
    InputTensorAdapterPtr hx_adapter;
    InputTensorAdapterPtr cx_adapter;

    using OutputTensorAdapterPtr = std::unique_ptr<RNN::Details::TransposeOutputTensorAdapter>;
    OutputTensorAdapterPtr y_adapter;
    OutputTensorAdapterPtr hy_adapter;
    OutputTensorAdapterPtr cy_adapter;

private:
    CudaGraphCompatibility graph_compatibility_;
};

}  // namespace nvidia_gpu
}  // namespace ov
