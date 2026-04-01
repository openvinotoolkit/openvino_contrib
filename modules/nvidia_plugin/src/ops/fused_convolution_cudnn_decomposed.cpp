// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_convolution_cudnn_decomposed.hpp"

#include <cudnn.h>

#include <openvino/core/except.hpp>
#include <ops/converters.hpp>

#include "cuda/constant_factory.hpp"

namespace ov {
namespace nvidia_gpu {

FusedConvolutionCuDnnDecomposed::FusedConvolutionCuDnnDecomposed(
    const CreationContext& context,
    const ov::Node& node,
    IndexCollection&& inputIds,
    IndexCollection&& outputIds,
    std::shared_ptr<Convolution::Details::ConvolutionDescriptorsCuDnn> convDescs,
    std::shared_ptr<CUDA::DnnTensorDescriptor> biasDesc,
    std::shared_ptr<CUDA::DnnTensorDescriptor> addDesc,
    std::shared_ptr<CUDA::DnnActivationDescriptor> activationDesc)
    : OperationCuDnn{context, node, std::move(inputIds), std::move(outputIds)},
      conv_descs_{convDescs},
      bias_desc_{biasDesc},
      activation_desc_{activationDesc},
      add_desc_{addDesc} {
    ThrowIfShouldNotDecompose();
}

void FusedConvolutionCuDnnDecomposed::Execute(const InferenceRequestContext& context,
                                              Inputs inputs,
                                              Outputs outputs,
                                              const Workbuffers& workbuffers) const {
    using ArgIndices = Convolution::Details::FusedConvolutionIndices;

    const bool includesOnlyBiasAdd = inputs.size() == 3;
    const bool includesSecondAddition = inputs.size() == 4;
    OPENVINO_ASSERT((includesOnlyBiasAdd && add_desc_ == nullptr) || (includesSecondAddition && add_desc_),
                    "Node name: ",
                    GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());

    const auto& dnnHandle = context.getThreadContext().dnnHandle();
    const void* onePtr = &CUDA::NumericConst<CUDA::constants::one>(conv_descs_->ElementType());
    const void* zeroPtr = &CUDA::NumericConst<CUDA::constants::zero>(conv_descs_->ElementType());
    void* workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    const auto& outDesc = conv_descs_->Output();
    auto outTensor = outputs[ArgIndices::output].get();

    throwIfError(::cudnnConvolutionForward(dnnHandle.get(),
                                           onePtr,
                                           conv_descs_->Input().get(),
                                           inputs[ArgIndices::input].get(),
                                           conv_descs_->Filter().get(),
                                           inputs[ArgIndices::filter].get(),
                                           conv_descs_->Conv().get(),
                                           conv_descs_->Algo().algo,
                                           workbuffer,
                                           conv_descs_->Algo().memory,
                                           zeroPtr,
                                           outDesc.get(),
                                           outTensor));
    throwIfError(::cudnnAddTensor(
        dnnHandle.get(), onePtr, bias_desc_->get(), inputs[ArgIndices::bias].get(), onePtr, outDesc.get(), outTensor));
    if (includesSecondAddition) {
        throwIfError(::cudnnAddTensor(dnnHandle.get(),
                                      onePtr,
                                      add_desc_->get(),
                                      inputs[ArgIndices::add].get(),
                                      onePtr,
                                      outDesc.get(),
                                      outTensor));
    }
    cudnnActivationMode_t mode;
    cudnnNanPropagation_t prop;
    double coef;
    throwIfError(::cudnnGetActivationDescriptor(activation_desc_->get(), &mode, &prop, &coef));
    if (mode != CUDNN_ACTIVATION_IDENTITY) {
        dnnHandle.activationForward(*activation_desc_, onePtr, outDesc, outTensor, zeroPtr, outDesc, outTensor);
    }
}

CudaGraphCompatibility FusedConvolutionCuDnnDecomposed::GetCudaGraphCompatibilityImpl() const {
    return CudaGraphCompatibility::FULL;
}

WorkbufferRequest FusedConvolutionCuDnnDecomposed::GetWorkBufferRequest() const {
    if (conv_descs_->Algo().memory != 0) {
        return {{}, {conv_descs_->Algo().memory}};
    } else {
        return {{}, {}};
    }
}

// cudnnConvolutionBiasActivationForward() doesn't work properly with CUDNN_ACTIVATION_IDENTITY and any algorithm
// other than CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, so we should decompose the convolution node and call
// separate cuDNN functions.
// For more information see:
// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
void FusedConvolutionCuDnnDecomposed::ThrowIfShouldNotDecompose() const {
    cudnnActivationMode_t mode;
    cudnnNanPropagation_t prop;
    double coef;
    throwIfError(::cudnnGetActivationDescriptor(activation_desc_->get(), &mode, &prop, &coef));
    if (mode != CUDNN_ACTIVATION_IDENTITY ||
        conv_descs_->Algo().algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) {
        throw_ov_exception(
            "ov::nvidia_gpu::FusedConvolutionCuDnnDecomposed: FusedConvolutionCuDnnDecomposed should only be used for "
            "CUDNN_ACTIVATION_IDENTITY and an algo other than CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM");
    }
}

}  // namespace nvidia_gpu
}  // namespace ov
