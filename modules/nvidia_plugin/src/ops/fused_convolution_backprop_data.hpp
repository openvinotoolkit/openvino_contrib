// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformer/nodes/fused_convolution_backprop_data.hpp>

#include "convolution_components/convolution_cudnn_components.hpp"
#include "cuda/dnn.hpp"
#include "cuda_operation_base.hpp"

namespace ov {
namespace nvidia_gpu {

class FusedConvolutionBackpropDataOp : public OperationCuDnn {
public:
    using NodeOp = ov::nvidia_gpu::nodes::FusedConvBackpropData;
    FusedConvolutionBackpropDataOp(const CreationContext& context,
                                   const NodeOp& node,
                                   IndexCollection&& inputIds,
                                   IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;
    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers& buffers) override;
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    static std::size_t GetBufferSize(const ov::Output<ov::Node>& output);

private:
    const Convolution::Details::FusedConvolutionBackwardDataParams params_;
    Convolution::Details::ConvolutionBackpropDataDescriptorCuDnn conv_descs_;
    std::shared_ptr<ov::Node> add_node_;
    gsl::span<const uint8_t, gsl::dynamic_extent> add_constant_;
    size_t conv_in_bytes_;
    size_t add_in_bytes_;
};

}  // namespace nvidia_gpu
}  // namespace ov
