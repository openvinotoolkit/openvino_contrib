// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory.h>

#include <openvino/op/group_conv.hpp>

#include "convolution_cudnn.hpp"
#include "cuda_operation_base.hpp"

namespace ov {
namespace nvidia_gpu {

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

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;
    WorkbufferRequest GetWorkBufferRequest() const override final;

private:
    ConvolutionCuDnn convolution_;
};

}  // namespace nvidia_gpu
}  // namespace ov
