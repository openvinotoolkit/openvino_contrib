// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_convolution.hpp"

#include <cuda_operation_registry.hpp>
#include <ngraph/partial_shape.hpp>

#include "convolution_components/convolution_components.hpp"

namespace CUDAPlugin {

GroupConvolutionOp::GroupConvolutionOp(const CreationContext &context,
                                       const NodeOp &node,
                                       IndexCollection &&inputIds,
                                       IndexCollection &&outputIds)
    : OperationCuDnn{context, node, move(inputIds), move(outputIds)},
      convolution_(context, node, move(inputIds), move(outputIds), Convolution::Details::ConvolutionParams{node}) {}

void GroupConvolutionOp::Execute(const InferenceRequestContext &context,
                                 Inputs inputTensors,
                                 Outputs outputTensors,
                                 const Workbuffers &buffers) const {
    convolution_.Execute(context, inputTensors, outputTensors, buffers);
}

WorkbufferRequest GroupConvolutionOp::GetWorkBufferRequest() const { return convolution_.GetWorkBufferRequest(); }

OPERATION_REGISTER(GroupConvolutionOp, GroupConvolution);

}  // namespace CUDAPlugin
