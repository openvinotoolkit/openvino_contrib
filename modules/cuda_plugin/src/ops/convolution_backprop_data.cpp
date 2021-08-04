// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_backprop_data.hpp"

#include <cuda_operation_registry.hpp>
#include "constant_factory.hpp"
#include "convolution_components.hpp"

namespace CUDAPlugin {

ConvolutionBackpropDataOp::ConvolutionBackpropDataOp(
    const CUDA::CreationContext& context,
    const NodeOp& node,
    IndexCollection&& inputIds,
    IndexCollection&& outputIds)
    : OperationCuDnn{context, node, std::move(inputIds), std::move(outputIds)}
    , descs_{ context, {{node}} } {
}

void ConvolutionBackpropDataOp::Execute(
    const InferenceRequestContext& context,
    Inputs inputs,
    Outputs outputs,
    const Workbuffers& workbuffers) {
    Expects(inputs.size() == 2 || inputs.size() == 3);
    Expects(outputs.size() == 1);
    void * workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    throwIfError(::cudnnConvolutionBackwardData(
        context.getThreadContext().dnnHandle().get(),
        &NumericConst<constants::one>(descs_.ElementType()),
        descs_.Filter().get(),
        inputs[ConvolutionBackpropDataOp::ArgIndices::filter].get(),
        descs_.dOutput().get(),
        inputs[ConvolutionBackpropDataOp::ArgIndices::doutput].get(),
        descs_.Conv().get(),
        descs_.Algo().algo,
        workbuffer,
        descs_.Algo().memory,
        &NumericConst<constants::zero>(descs_.ElementType()),
        descs_.dInput().get(),
        outputs[ConvolutionBackpropDataOp::ArgIndices::dinput].get()));
}

OPERATION_REGISTER(ConvolutionBackpropDataOp, ConvolutionBackpropData);

} // namespace CUDAPlugin
