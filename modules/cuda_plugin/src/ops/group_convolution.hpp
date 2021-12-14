// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory.h>

#include <ngraph/op/group_conv.hpp>

#include "cuda_operation_base.hpp"
#include "ops/convolution_cudnn.hpp"

namespace CUDAPlugin {

class GroupConvolutionOp : public OperationCuDnn {
public:
    using NodeOp = ngraph::op::v1::GroupConvolution;
    GroupConvolutionOp(const CreationContext& context,
                       const NodeOp& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const final;
    WorkbufferRequest GetWorkBufferRequest() const final;

private:
    ConvolutionCuDnn convolution_;
};

}  // namespace CUDAPlugin
