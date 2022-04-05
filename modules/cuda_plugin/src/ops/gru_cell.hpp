// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <ngraph/node.hpp>

#include "rnn_components/rnn_components.hpp"
#include "rnn_components/rnn_cudnn_components.hpp"

namespace CUDAPlugin {

/**
 * @brief Implements `ngraph::op::v4::GRUCell` using cuDNN API
 */
class GRUCellOp : public OperationCuDnn {
public:
    GRUCellOp(const CreationContext& context,
              const ngraph::Node& node,
              IndexCollection&& inputIds,
              IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override;
    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override;
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    RNN::Details::GRUCellParams params_;
    RNN::Details::GRUCellDescriptorsCuDnn descs_;
};

}  // namespace CUDAPlugin
