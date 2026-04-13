// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_convolution.hpp"

#include <cuda_operation_registry.hpp>

#include "convolution_components/convolution_components.hpp"

namespace ov {
namespace nvidia_gpu {

GroupConvolutionOp::GroupConvolutionOp(const CreationContext &context,
                                       const NodeOp &node,
                                       IndexCollection &&inputIds,
                                       IndexCollection &&outputIds)
    : OperationCuDnn{context, node, move(inputIds), move(outputIds)},
      convolution_(context, node, {}, {}, Convolution::Details::ConvolutionParams{node}) {}

void GroupConvolutionOp::Execute(const InferenceRequestContext &context,
                                 Inputs inputTensors,
                                 Outputs outputTensors,
                                 const Workbuffers &buffers) const {
    convolution_.Execute(context, inputTensors, outputTensors, buffers);
}

CudaGraphCompatibility GroupConvolutionOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

WorkbufferRequest GroupConvolutionOp::GetWorkBufferRequest() const { return convolution_.GetWorkBufferRequest(); }

OPERATION_REGISTER(GroupConvolutionOp, GroupConvolution);

}  // namespace nvidia_gpu
}  // namespace ov
