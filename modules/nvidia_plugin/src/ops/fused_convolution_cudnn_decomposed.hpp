// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "convolution_components/convolution_cudnn_components.hpp"
#include "cuda/dnn.hpp"
#include "cuda_operation_base.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief This class was created as a workaround for the following cuDNN behavior:
 * cudnnConvolutionBiasActivationForward() doesn't work properly with CUDNN_ACTIVATION_IDENTITY and any algorithm
 * other than CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, so we should decompose the convolution node and call
 * separate cuDNN functions.
 * For more information see:
 * https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
 */
class FusedConvolutionCuDnnDecomposed : public OperationCuDnn {
public:
    FusedConvolutionCuDnnDecomposed(const CreationContext& context,
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

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;
    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override {}
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    void ThrowIfShouldNotDecompose() const;

    WorkbufferIds workbuffer_ids_;
    std::shared_ptr<Convolution::Details::ConvolutionDescriptorsCuDnn> conv_descs_;
    std::shared_ptr<CUDA::DnnTensorDescriptor> bias_desc_;
    std::shared_ptr<CUDA::DnnTensorDescriptor> add_desc_;
    std::shared_ptr<CUDA::DnnActivationDescriptor> activation_desc_;
};

}  // namespace nvidia_gpu
}  // namespace ov
