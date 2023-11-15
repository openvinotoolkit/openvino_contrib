// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_components/convolution_cudnn_components.hpp"
#include "cuda_operation_base.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Implements `ov::op::v1::Convolution` using cuDNN API
 * which doesn't support asymmetric padding.
 */
class ConvolutionCuDnn : public OperationCuDnn {
public:
    ConvolutionCuDnn(const CreationContext& context,
                     const ov::Node& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds,
                     const Convolution::Details::ConvolutionParams& params);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override;

    WorkbufferRequest GetWorkBufferRequest() const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;

private:
    Convolution::Details::ConvolutionDescriptorsCuDnn descs_;
};

}  // namespace nvidia_gpu
}  // namespace ov
