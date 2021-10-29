// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <ngraph/node.hpp>

#include "lstm_sequence_components.hpp"
#include "lstm_sequence_cudnn_components.hpp"

namespace CUDAPlugin {

class LSTMSequenceOpBase : public OperationCuDnn {
public:
    using NodeOp = ngraph::op::v5::LSTMSequence;
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

    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override;
    WorkbufferRequest GetWorkBufferRequest() const override;

protected:
    const RNN::Details::LSTMSequenceParamsCuDnn params_;
    RNN::Details::LSTMSequenceDescriptorsCuDnn descs_;

    using WorkbufferDesc = RNN::Details::WorkbufferDesc;
    mutable WorkbufferDesc ib_seq_lengths_;
    mutable WorkbufferDesc ib_weight_space_;
    mutable WorkbufferDesc mb_work_space_;

    using InputTensorAdapterPtr = std::unique_ptr<RNN::Details::TransposeInputTensorAdapter>;
    InputTensorAdapterPtr x_adapter;
    InputTensorAdapterPtr hx_adapter;
    InputTensorAdapterPtr cx_adapter;

    using OutputTensorAdapterPtr = std::unique_ptr<RNN::Details::TransposeOutputTensorAdapter>;
    OutputTensorAdapterPtr y_adapter;
    OutputTensorAdapterPtr hy_adapter;
    OutputTensorAdapterPtr cy_adapter;
};

}  // namespace CUDAPlugin
