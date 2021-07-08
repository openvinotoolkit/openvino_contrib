// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <memory>
#include <transformer/nodes/convolution2d_biasadd_activation.hpp>
#include "convolution_components.hpp"

namespace CUDAPlugin {

class Convolution2DBiasAddActivationOp : public OperationCuDnn {
public:
    using NodeOp = CUDAPlugin::nodes::Conv2DBiasAddActivation;
    Convolution2DBiasAddActivationOp(const NodeOp& node, IndexCollection&& inputIds,
                                     IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context, Inputs inputTensors,
                 Outputs outputTensors, const Workbuffers& workbuffers) override;
    WorkbufferRequest GetWorkBufferRequest() const override;

    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override {}
    const WorkbufferIndices& GetWorkbufferIds() const override;
    WorkbufferStatus SetWorkbufferIds(WorkbufferIndices&& workbufferIds) override;

    using ArgIndices = Convolution::Details::ConvolutionBiasAddActivationIndices;

private:
    void CreateImpl(const NodeOp& node);
    std::unique_ptr<IOperationExec> impl_;
};

}  // namespace CUDAPlugin
