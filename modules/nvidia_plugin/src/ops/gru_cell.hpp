// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "rnn_components/rnn_components.hpp"
#include "rnn_components/rnn_cudnn_components.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Implements `ov::op::v4::GRUCell` using cuDNN API
 */
class GRUCellOp : public OperationCuDnn {
public:
    GRUCellOp(const CreationContext& context,
              const ov::Node& node,
              IndexCollection&& inputIds,
              IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;
    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override;
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    RNN::Details::GRUCellParams params_;
    RNN::Details::GRUCellDescriptorsCuDnn descs_;
};

}  // namespace nvidia_gpu
}  // namespace ov
