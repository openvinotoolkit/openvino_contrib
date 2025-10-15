// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "convolution_components/convolution_components.hpp"
#include "convolution_components/convolution_cudnn_components.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Implements both `ov::op::v1::ConvolutionBackpropData`
 * and `ov::op::v1::GroupConvolutionBackpropData` using cuDNN API
 * which doesn't support asymmetric padding.
 */
template <typename T>
class ConvBackpropDataOp : public OperationCuDnn {
public:
    using NodeOp = T;

    ConvBackpropDataOp(const CreationContext& context,
                       const NodeOp& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;

    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override;
    WorkbufferRequest GetWorkBufferRequest() const override;

    using ArgIndices = Convolution::Details::ConvBackArgIndices;

private:
    Convolution::Details::ConvolutionBackpropDataDescriptorCuDnn descs_;
};

template <typename T>
inline void ConvBackpropDataOp<T>::InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) {}

template <typename T>
inline WorkbufferRequest ConvBackpropDataOp<T>::GetWorkBufferRequest() const {
    if (descs_.Algo().memory != 0) {
        return {{}, {descs_.Algo().memory}};
    } else {
        return {{}, {}};
    }
}

class ConvolutionBackpropDataOp : public ConvBackpropDataOp<ov::op::v1::ConvolutionBackpropData> {
public:
    using ConvBackpropDataOp::ConvBackpropDataOp;
};
class GroupConvolutionBackpropDataOp : public ConvBackpropDataOp<ov::op::v1::GroupConvolutionBackpropData> {
public:
    using ConvBackpropDataOp::ConvBackpropDataOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
