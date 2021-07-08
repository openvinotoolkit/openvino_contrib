// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gsl/gsl_assert>
#include <details/ie_exception.hpp>

#include "convolution_cudnn.hpp"
#include "convolution.hpp"
#include "constant_factory.hpp"
#include <cudnn.h>

namespace CUDAPlugin {

ConvolutionCuDnn::ConvolutionCuDnn(const Convolution::Details::ConvolutionParams& params)
    : descs_{ params }  {
}

void ConvolutionCuDnn::Execute(const InferenceRequestContext& context,
                               Inputs inputs, Outputs outputs,
                               const Workbuffers& workbuffers) {
    Expects(inputs.size() == 2);
    Expects(outputs.size() == 1);
    void * workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    cudnnStatus_t status = ::cudnnConvolutionForward(
                                context.getThreadContext().dnnHandle().get(),
                                &NumericConst<constants::one>(descs_.ElementType()),
                                descs_.Input().get(),
                                inputs[ConvolutionOp::ArgIndices::input].get(),
                                descs_.Filter().get(),
                                inputs[ConvolutionOp::ArgIndices::filter].get(),
                                descs_.Conv().get(),
                                descs_.Algo().algo,
                                workbuffer,
                                descs_.Algo().memory,
                                &NumericConst<constants::zero>(descs_.ElementType()),
                                descs_.Output().get(),
                                outputs[ConvolutionOp::ArgIndices::output].get());
    throwIfError(status);
}

WorkbufferRequest ConvolutionCuDnn::GetWorkBufferRequest() const {
    if (descs_.Algo().memory != 0)
      return {{}, {descs_.Algo().memory}};
    else
      return {{}, {}};
}

} // namespace CUDAPlugin
