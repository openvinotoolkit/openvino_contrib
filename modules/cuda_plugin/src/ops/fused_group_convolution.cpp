// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_group_convolution.hpp"

#include <fmt/format.h>

#include <gsl/gsl_assert>
#include <optional>

#include "cuda_operation_registry.hpp"
#include "fused_convolution_cudnn.hpp"

namespace CUDAPlugin {

FusedGroupConvolutionOp::FusedGroupConvolutionOp(const CreationContext& context,
                                                 const NodeOp& op,
                                                 IndexCollection&& inputIds,
                                                 IndexCollection&& outputIds)
    : OperationCuDnn(context, op, std::move(inputIds), std::move(outputIds)), fused_conv_{context, op, {}, {}} {}

void FusedGroupConvolutionOp::Execute(const InferenceRequestContext& context,
                                      Inputs inputs,
                                      Outputs outputs,
                                      const Workbuffers& workbuffers) const {
    fused_conv_.Execute(context, inputs, outputs, workbuffers);
}

WorkbufferRequest FusedGroupConvolutionOp::GetWorkBufferRequest() const { return fused_conv_.GetWorkBufferRequest(); }

const WorkbufferIds& FusedGroupConvolutionOp::GetWorkbufferIds() const { return fused_conv_.GetWorkbufferIds(); }

IOperationExec::WorkbufferStatus FusedGroupConvolutionOp::SetWorkbufferIds(WorkbufferIds&& workbufferIds) {
    return fused_conv_.SetWorkbufferIds(std::move(workbufferIds));
}

OPERATION_REGISTER(FusedGroupConvolutionOp, FusedGroupConvolution);

}  // namespace CUDAPlugin
