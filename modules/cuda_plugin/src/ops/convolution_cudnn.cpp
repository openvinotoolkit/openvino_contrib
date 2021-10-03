// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_cudnn.hpp"

#include <cudnn.h>

#include <details/ie_exception.hpp>
#include <gsl/gsl_assert>

#include "constant_factory.hpp"

namespace CUDAPlugin {

ConvolutionCuDnn::ConvolutionCuDnn(const CreationContext& context,
                                   const ngraph::Node& node,
                                   IndexCollection&& inputIds,
                                   IndexCollection&& outputIds,
                                   const Convolution::Details::ConvolutionParams& params)
    : OperationCuDnn{context, node, move(inputIds), move(outputIds)}, descs_{context, params} {}

void ConvolutionCuDnn::Execute(const InferenceRequestContext& context,
                               Inputs inputs,
                               Outputs outputs,
                               const Workbuffers& workbuffers) const {
    Expects(inputs.size() == 2);
    Expects(outputs.size() == 1);
    void* workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    cudnnStatus_t status = ::cudnnConvolutionForward(context.getThreadContext().dnnHandle().get(),
                                                     &NumericConst<constants::one>(descs_.ElementType()),
                                                     descs_.Input().get(),
                                                     inputs[Convolution::Details::ConvArgIndices::input].get(),
                                                     descs_.Filter().get(),
                                                     inputs[Convolution::Details::ConvArgIndices::filter].get(),
                                                     descs_.Conv().get(),
                                                     descs_.Algo().algo,
                                                     workbuffer,
                                                     descs_.Algo().memory,
                                                     &NumericConst<constants::zero>(descs_.ElementType()),
                                                     descs_.Output().get(),
                                                     outputs[Convolution::Details::ConvArgIndices::output].get());
    throwIfError(status);
}

WorkbufferRequest ConvolutionCuDnn::GetWorkBufferRequest() const {
    if (descs_.Algo().memory != 0)
        return {{}, {descs_.Algo().memory}};
    else
        return {{}, {}};
}

}  // namespace CUDAPlugin
