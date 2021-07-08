// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gsl/gsl_assert>
#include <details/ie_exception.hpp>

#include "convolution2d_biasadd_activation_cudnn.hpp"
#include "convolution2d_biasadd_activation.hpp"
#include "converters.hpp"
#include "constant_factory.hpp"
#include <cudnn.h>

namespace CUDAPlugin {

Convolution2DBiasAddActivationCuDnn::Convolution2DBiasAddActivationCuDnn(
    const Convolution::Details::ConvolutionBiasAddActivationParams& params)
    : conv_descs_{ params.conv_ }
    , bias_desc_{ MakeBiasDescriptor(params.bias_shape_, params.conv_.element_type_) }
    , activation_desc_{ MakeActivationDescriptor(params.activation_) } {
}

void Convolution2DBiasAddActivationCuDnn::Execute(const InferenceRequestContext& context,
                                                  Inputs inputs, Outputs outputs,
                                                  const Workbuffers& workbuffers) {
    Expects(inputs.size() == 3);
    Expects(outputs.size() == 1);
    void * workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    using ArgIndices = Convolution2DBiasAddActivationOp::ArgIndices;
    cudnnStatus_t status = ::cudnnConvolutionBiasActivationForward(
                                context.getThreadContext().dnnHandle().get(),
                                &NumericConst<constants::one>(conv_descs_.ElementType()),
                                conv_descs_.Input().get(),
                                inputs[ArgIndices::input].get(),
                                conv_descs_.Filter().get(),
                                inputs[ArgIndices::filter].get(),
                                conv_descs_.Conv().get(),
                                conv_descs_.Algo().algo,
                                workbuffer,
                                conv_descs_.Algo().memory,
                                &NumericConst<constants::zero>(conv_descs_.ElementType()),
                                conv_descs_.Output().get(),
                                outputs[ArgIndices::output].get(),
                                bias_desc_.get(),
                                inputs[ArgIndices::bias].get(),
                                activation_desc_.get(),
                                conv_descs_.Output().get(),
                                outputs[ArgIndices::output].get());
    throwIfError(status);
}

WorkbufferRequest
Convolution2DBiasAddActivationCuDnn::GetWorkBufferRequest() const {
    if (conv_descs_.Algo().memory != 0)
      return {{}, {conv_descs_.Algo().memory}};
    else
      return {{}, {}};
}

CUDA::DnnTensorDescriptor
Convolution2DBiasAddActivationCuDnn::MakeBiasDescriptor(const ngraph::Shape& shape,
                                                      ngraph::element::Type_t element_type) {
    std::array<int, CUDNN_DIM_MAX> int_shape;
    std::copy(shape.begin(), shape.end(), int_shape.begin());
    return CUDA::DnnTensorDescriptor {
        cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        convertDataType<cudnnDataType_t>(element_type),
        static_cast<int>(shape.size()),
        int_shape.data()
    };
}

CUDA::DnnActivationDescriptor
Convolution2DBiasAddActivationCuDnn::MakeActivationDescriptor(nodes::ActivationMode mode) {
    return CUDA::DnnActivationDescriptor {
        convertActivationMode(mode),
        CUDNN_PROPAGATE_NAN, 0
    };
}

} // namespace CUDAPlugin
