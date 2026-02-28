// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_backprop_data.hpp"

#include <cuda_operation_registry.hpp>

#include "convolution_components/convolution_components.hpp"
#include "cuda/constant_factory.hpp"

namespace ov {
namespace nvidia_gpu {

template <typename T>
ConvBackpropDataOp<T>::ConvBackpropDataOp(const CreationContext& context,
                                          const NodeOp& node,
                                          IndexCollection&& inputIds,
                                          IndexCollection&& outputIds)
    : OperationCuDnn{context, node, std::move(inputIds), std::move(outputIds)}, descs_{context, {{node}}} {}

template <typename T>
void ConvBackpropDataOp<T>::Execute(const InferenceRequestContext& context,
                                    Inputs inputs,
                                    Outputs outputs,
                                    const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(inputs.size() == 2 || inputs.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    void* workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    throwIfError(::cudnnConvolutionBackwardData(context.getThreadContext().dnnHandle().get(),
                                                &CUDA::NumericConst<CUDA::constants::one>(descs_.ElementType()),
                                                descs_.Filter().get(),
                                                inputs[ConvBackpropDataOp::ArgIndices::filter].get(),
                                                descs_.dOutput().get(),
                                                inputs[ConvBackpropDataOp::ArgIndices::doutput].get(),
                                                descs_.Conv().get(),
                                                descs_.Algo().algo,
                                                workbuffer,
                                                descs_.Algo().memory,
                                                &CUDA::NumericConst<CUDA::constants::zero>(descs_.ElementType()),
                                                descs_.dInput().get(),
                                                outputs[ConvBackpropDataOp::ArgIndices::dinput].get()));
}

template <typename T>
CudaGraphCompatibility ConvBackpropDataOp<T>::GetCudaGraphCompatibilityImpl() const {
    return CudaGraphCompatibility::FULL;
}

OPERATION_REGISTER(ConvolutionBackpropDataOp, ConvolutionBackpropData);
OPERATION_REGISTER(GroupConvolutionBackpropDataOp, GroupConvolutionBackpropData);

}  // namespace nvidia_gpu
}  // namespace ov
