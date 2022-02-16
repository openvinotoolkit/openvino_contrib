// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <transformer/nodes/fused_convolution.hpp>

#include "fused_convolution_cudnn.hpp"

namespace CUDAPlugin {

class FusedGroupConvolutionCuDnnOp : public OperationCuDnn {
public:
    using NodeOp = nodes::FusedGroupConvolution;

    FusedGroupConvolutionCuDnnOp(const CreationContext& context,
                                 const NodeOp& op,
                                 IndexCollection&& inputIds,
                                 IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    WorkbufferRequest GetWorkBufferRequest() const override;

    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override {}
    const WorkbufferIds& GetWorkbufferIds() const override;
    WorkbufferStatus SetWorkbufferIds(WorkbufferIds&& workbufferIds) override;

private:
    FusedConvolutionCuDnn fused_conv_;
};

}  // namespace CUDAPlugin
