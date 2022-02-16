// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_group_convolution.hpp"

#include <gsl/gsl_assert>

#include "cuda_operation_registry.hpp"
#include "fused_convolution_cudnn.hpp"

namespace CUDAPlugin {

FusedGroupConvolutionCuDnnOp::FusedGroupConvolutionCuDnnOp(const CreationContext& context,
                                                           const NodeOp& node,
                                                           IndexCollection&& inputIds,
                                                           IndexCollection&& outputIds)
    : OperationCuDnn(context, node, std::move(inputIds), std::move(outputIds)),
      fused_conv_{context, node, {}, {}, Convolution::Details::FusedConvolutionParams{node}} {}

void FusedGroupConvolutionCuDnnOp::Execute(const InferenceRequestContext& context,
                                           Inputs inputs,
                                           Outputs outputs,
                                           const Workbuffers& workbuffers) const {
    fused_conv_.Execute(context, inputs, outputs, workbuffers);
}

WorkbufferRequest FusedGroupConvolutionCuDnnOp::GetWorkBufferRequest() const { return fused_conv_.GetWorkBufferRequest(); }

const WorkbufferIds& FusedGroupConvolutionCuDnnOp::GetWorkbufferIds() const { return fused_conv_.GetWorkbufferIds(); }

IOperationExec::WorkbufferStatus FusedGroupConvolutionCuDnnOp::SetWorkbufferIds(WorkbufferIds&& workbufferIds) {
    return fused_conv_.SetWorkbufferIds(std::move(workbufferIds));
}

OPERATION_REGISTER(FusedGroupConvolutionCuDnnOp, FusedGroupConvolution);

}  // namespace CUDAPlugin
