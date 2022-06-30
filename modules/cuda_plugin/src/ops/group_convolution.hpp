// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory.h>

#include <openvino/op/group_conv.hpp>

#include "convolution_cudnn.hpp"
#include "cuda_operation_base.hpp"

namespace CUDAPlugin {

class GroupConvolutionOp : public OperationCuDnn {
public:
    using NodeOp = ov::op::v1::GroupConvolution;
    GroupConvolutionOp(const CreationContext& context,
                       const NodeOp& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override final;
    WorkbufferRequest GetWorkBufferRequest() const override final;

private:
    ConvolutionCuDnn convolution_;
};

}  // namespace CUDAPlugin
