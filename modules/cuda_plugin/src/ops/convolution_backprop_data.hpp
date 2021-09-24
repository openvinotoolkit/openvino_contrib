// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "convolution_components.hpp"
#include "convolution_cudnn_components.hpp"

namespace CUDAPlugin {

/**
 * @brief Implements `ngraph::op::v1::Convolution` using cuDNN API
 * which doesn't support asymmetric padding.
 */
class ConvolutionBackpropDataOp : public OperationCuDnn {
public:
    using NodeOp = ngraph::op::v1::ConvolutionBackpropData;
    ConvolutionBackpropDataOp(const CUDA::CreationContext& context,
                              const NodeOp& node,
                              IndexCollection&& inputIds,
                              IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override;
    WorkbufferRequest GetWorkBufferRequest() const override;

    using ArgIndices = Convolution::Details::ConvBackArgIndices;

private:
    Convolution::Details::ConvolutionBackpropDataDescriptorCuDnn descs_;
};

inline void ConvolutionBackpropDataOp::InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) {}

inline WorkbufferRequest ConvolutionBackpropDataOp::GetWorkBufferRequest() const {
    if (descs_.Algo().memory != 0) {
        return {{}, {descs_.Algo().memory}};
    } else {
        return {{}, {}};
    }
}

}  // namespace CUDAPlugin
