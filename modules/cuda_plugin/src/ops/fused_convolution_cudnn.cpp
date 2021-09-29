// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_convolution_cudnn.hpp"

#include <cudnn.h>

#include <details/ie_exception.hpp>
#include <gsl/gsl_assert>

#include "constant_factory.hpp"
#include "converters.hpp"
#include "fused_convolution.hpp"

namespace CUDAPlugin {

FusedConvolutionCuDnn::FusedConvolutionCuDnn(const CreationContext& context,
                                             const Convolution::Details::FusedConvolutionParams& params)
    : conv_descs_{context, params.conv_},
      bias_desc_{MakeBiasDescriptor(params.bias_shape_, params.conv_.element_type_)},
      activation_desc_{MakeActivationDescriptor(params.activation_)} {
    if (params.add_shape_) {
        add_desc_ = MakeBiasDescriptor(params.add_shape_.value(), params.conv_.element_type_);
    }
}

void FusedConvolutionCuDnn::Execute(const InferenceRequestContext& context,
                                    Inputs inputs,
                                    Outputs outputs,
                                    const Workbuffers& workbuffers) const {
    using ArgIndices = FusedConvolutionOp::ArgIndices;

    const bool includesOnlyBiasAdd = inputs.size() == 3;
    const bool includesSecondAddition = inputs.size() == 4;
    Expects(includesOnlyBiasAdd || includesSecondAddition);
    Expects(outputs.size() == 1);
    void* workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    const auto& dnnHandle = context.getThreadContext().dnnHandle();

    const constants::AnyNumeric* alpha2 = nullptr;
    cudnnTensorDescriptor_t zTensorDesc;
    const void* zTensorIn = nullptr;
    if (includesOnlyBiasAdd) {
        alpha2 = &NumericConst<constants::zero>(conv_descs_.ElementType());
        zTensorDesc = conv_descs_.Output().get();
        zTensorIn = outputs[ArgIndices::output].get();
    } else {
        alpha2 = &NumericConst<constants::one>(conv_descs_.ElementType());
        zTensorDesc = add_desc_.value().get();
        zTensorIn = inputs[ArgIndices::add].get();
    }
    throwIfError(::cudnnConvolutionBiasActivationForward(dnnHandle.get(),
                                                         &NumericConst<constants::one>(conv_descs_.ElementType()),
                                                         conv_descs_.Input().get(),
                                                         inputs[ArgIndices::input].get(),
                                                         conv_descs_.Filter().get(),
                                                         inputs[ArgIndices::filter].get(),
                                                         conv_descs_.Conv().get(),
                                                         conv_descs_.Algo().algo,
                                                         workbuffer,
                                                         conv_descs_.Algo().memory,
                                                         alpha2,
                                                         zTensorDesc,
                                                         zTensorIn,
                                                         bias_desc_.get(),
                                                         inputs[ArgIndices::bias].get(),
                                                         activation_desc_.get(),
                                                         conv_descs_.Output().get(),
                                                         outputs[ArgIndices::output].get()));
}

WorkbufferRequest FusedConvolutionCuDnn::GetWorkBufferRequest() const {
    if (conv_descs_.Algo().memory != 0)
        return {{}, {conv_descs_.Algo().memory}};
    else
        return {{}, {}};
}

CUDA::DnnTensorDescriptor FusedConvolutionCuDnn::MakeBiasDescriptor(const ngraph::Shape& shape,
                                                                    ngraph::element::Type_t element_type) {
    std::array<int, CUDNN_DIM_MAX> int_shape;
    std::copy(shape.begin(), shape.end(), int_shape.begin());
    return CUDA::DnnTensorDescriptor{cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                                     convertDataType<cudnnDataType_t>(element_type),
                                     static_cast<int>(shape.size()),
                                     int_shape.data()};
}

CUDA::DnnActivationDescriptor FusedConvolutionCuDnn::MakeActivationDescriptor(nodes::ActivationMode mode) {
    return CUDA::DnnActivationDescriptor{convertActivationMode(mode), CUDNN_PROPAGATE_NAN, 0};
}

}  // namespace CUDAPlugin
