// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_cudnn.hpp"

#include <cudnn.h>

#include <openvino/core/except.hpp>

#include "cuda/constant_factory.hpp"

namespace ov {
namespace nvidia_gpu {

ConvolutionCuDnn::ConvolutionCuDnn(const CreationContext& context,
                                   const ov::Node& node,
                                   IndexCollection&& inputIds,
                                   IndexCollection&& outputIds,
                                   const Convolution::Details::ConvolutionParams& params)
    : OperationCuDnn{context, node, move(inputIds), move(outputIds)}, descs_{context, params} {}

void ConvolutionCuDnn::Execute(const InferenceRequestContext& context,
                               Inputs inputs,
                               Outputs outputs,
                               const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(inputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    void* workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    cudnnStatus_t status = ::cudnnConvolutionForward(context.getThreadContext().dnnHandle().get(),
                                                     &CUDA::NumericConst<CUDA::constants::one>(descs_.ElementType()),
                                                     descs_.Input().get(),
                                                     inputs[Convolution::Details::ConvArgIndices::input].get(),
                                                     descs_.Filter().get(),
                                                     inputs[Convolution::Details::ConvArgIndices::filter].get(),
                                                     descs_.Conv().get(),
                                                     descs_.Algo().algo,
                                                     workbuffer,
                                                     descs_.Algo().memory,
                                                     &CUDA::NumericConst<CUDA::constants::zero>(descs_.ElementType()),
                                                     descs_.Output().get(),
                                                     outputs[Convolution::Details::ConvArgIndices::output].get());
    throwIfError(status);
}

CudaGraphCompatibility ConvolutionCuDnn::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::FULL; }

WorkbufferRequest ConvolutionCuDnn::GetWorkBufferRequest() const {
    if (descs_.Algo().memory != 0)
        return {{}, {descs_.Algo().memory}};
    else
        return {{}, {}};
}
}  // namespace nvidia_gpu
}  // namespace ov
