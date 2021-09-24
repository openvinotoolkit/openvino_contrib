// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <memory>
#include <transformer/nodes/fused_convolution.hpp>

#include "convolution_components.hpp"

namespace CUDAPlugin {

class FusedConvolutionOp : public OperationCuDnn {
public:
    using NodeOp = CUDAPlugin::nodes::FusedConvolution;
    FusedConvolutionOp(const CUDA::CreationContext& context,
                       const NodeOp& node,
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

    using ArgIndices = Convolution::Details::FusedConvolutionIndices;

private:
    void CreateImpl(const CUDA::CreationContext& context, const NodeOp& node);
    std::unique_ptr<IOperationExec> impl_;
};

}  // namespace CUDAPlugin
