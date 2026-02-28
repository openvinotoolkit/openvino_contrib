// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_convolution_cudnn.hpp"

#include <cudnn.h>

#include <openvino/core/except.hpp>
#include <ops/converters.hpp>

#include "cuda/constant_factory.hpp"
#include "transformer/nodes/activation_type.hpp"

namespace ov {
namespace nvidia_gpu {

FusedConvolutionCuDnn::FusedConvolutionCuDnn(const CreationContext& context,
                                             const ov::Node& node,
                                             IndexCollection&& inputIds,
                                             IndexCollection&& outputIds,
                                             Convolution::Details::FusedConvolutionParams params)
    : OperationCuDnn{context, node, std::move(inputIds), std::move(outputIds)},
      conv_descs_{std::make_shared<Convolution::Details::ConvolutionDescriptorsCuDnn>(context, params.conv_,
        std::vector<cudnnDataType_t>{CUDNN_DATA_HALF, CUDNN_DATA_FLOAT})}, // 119703: investigate whether we need HALF here
      bias_desc_{Convolution::Details::MakeFusedAddDescriptor(params.bias_shape_, params.conv_.element_type_)},
      add_desc_{params.add_shape_ ? Convolution::Details::MakeFusedAddDescriptor(params.add_shape_.value(),
                                                                                 params.conv_.element_type_)
                                  : nullptr},
      activation_desc_{Convolution::Details::MakeFusedActivationDescriptor(params.activation_)} {
    ThrowIfShouldDecompose();
}

FusedConvolutionCuDnn::FusedConvolutionCuDnn(
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
      add_desc_{addDesc},
      activation_desc_{activationDesc} {
    ThrowIfShouldDecompose();
}

void FusedConvolutionCuDnn::Execute(const InferenceRequestContext& context,
                                    Inputs inputs,
                                    Outputs outputs,
                                    const Workbuffers& workbuffers) const {
    using ArgIndices = Convolution::Details::FusedConvolutionIndices;

    const bool includesOnlyBiasAdd = inputs.size() == 3;
    const bool includesSecondAddition = inputs.size() == 4;
    OPENVINO_ASSERT(includesOnlyBiasAdd || includesSecondAddition, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    void* workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    const auto& dnnHandle = context.getThreadContext().dnnHandle();

    const CUDA::constants::AnyNumeric* alpha2 = nullptr;
    cudnnTensorDescriptor_t zTensorDesc;
    const void* zTensorIn = nullptr;
    if (includesOnlyBiasAdd) {
        alpha2 = &CUDA::NumericConst<CUDA::constants::zero>(conv_descs_->DescType());
        zTensorDesc = conv_descs_->Output().get();
        zTensorIn = outputs[ArgIndices::output].get();
    } else {
        alpha2 = &CUDA::NumericConst<CUDA::constants::one>(conv_descs_->DescType());
        zTensorDesc = add_desc_->get();
        zTensorIn = inputs[ArgIndices::add].get();
    }

    throwIfError(
        ::cudnnConvolutionBiasActivationForward(dnnHandle.get(),
                                                &CUDA::NumericConst<CUDA::constants::one>(conv_descs_->DescType()),
                                                conv_descs_->Input().get(),
                                                inputs[ArgIndices::input].get(),
                                                conv_descs_->Filter().get(),
                                                inputs[ArgIndices::filter].get(),
                                                conv_descs_->Conv().get(),
                                                conv_descs_->Algo().algo,
                                                workbuffer,
                                                conv_descs_->Algo().memory,
                                                alpha2,
                                                zTensorDesc,
                                                zTensorIn,
                                                bias_desc_->get(),
                                                inputs[ArgIndices::bias].get(),
                                                activation_desc_->get(),
                                                conv_descs_->Output().get(),
                                                outputs[ArgIndices::output].get()));
}

CudaGraphCompatibility FusedConvolutionCuDnn::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

WorkbufferRequest FusedConvolutionCuDnn::GetWorkBufferRequest() const {
    if (conv_descs_->Algo().memory != 0)
        return {{}, {conv_descs_->Algo().memory}};
    else
        return {{}, {}};
}

// cudnnConvolutionBiasActivationForward() doesn't work properly with CUDNN_ACTIVATION_IDENTITY and any algorithm
// other than CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, so we should decompose the convolution node and call
// separate cuDNN functions.
// For more information see:
// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
void FusedConvolutionCuDnn::ThrowIfShouldDecompose() const {
    cudnnActivationMode_t mode;
    cudnnNanPropagation_t prop;
    double coef;
    throwIfError(::cudnnGetActivationDescriptor(activation_desc_->get(), &mode, &prop, &coef));
    if (mode == CUDNN_ACTIVATION_IDENTITY &&
        conv_descs_->Algo().algo != CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) {
        throw_ov_exception(
            "ov::nvidia_gpu::FusedConvolutionCuDnn: CUDNN_ACTIVATION_IDENTITY can't be used with "
            "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM");
    }
}

}  // namespace nvidia_gpu
}  // namespace ov
