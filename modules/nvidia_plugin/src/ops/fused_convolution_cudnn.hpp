// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "convolution_components/convolution_cudnn_components.hpp"
#include "cuda/dnn.hpp"
#include "cuda_operation_base.hpp"

namespace ov {
namespace nvidia_gpu {

class FusedConvolutionCuDnn : public OperationCuDnn {
public:
    FusedConvolutionCuDnn(const CreationContext& context,
                          const ov::Node& node,
                          IndexCollection&& inputIds,
                          IndexCollection&& outputIds,
                          Convolution::Details::FusedConvolutionParams params);

    FusedConvolutionCuDnn(const CreationContext& context,
                          const ov::Node& node,
                          IndexCollection&& inputIds,
                          IndexCollection&& outputIds,
                          std::shared_ptr<Convolution::Details::ConvolutionDescriptorsCuDnn> conv_descs_,
                          std::shared_ptr<CUDA::DnnTensorDescriptor> bias_desc_,
                          std::shared_ptr<CUDA::DnnTensorDescriptor> add_desc_,
                          std::shared_ptr<CUDA::DnnActivationDescriptor> activation_desc_);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;
    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override {}
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    void ThrowIfShouldDecompose() const;
    WorkbufferIds workbuffer_ids_;
    std::shared_ptr<Convolution::Details::ConvolutionDescriptorsCuDnn> conv_descs_;
    std::shared_ptr<CUDA::DnnTensorDescriptor> bias_desc_;
    std::shared_ptr<CUDA::DnnTensorDescriptor> add_desc_;
    std::shared_ptr<CUDA::DnnActivationDescriptor> activation_desc_;
};

}  // namespace nvidia_gpu
}  // namespace ov
