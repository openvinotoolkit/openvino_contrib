// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_cudnn_components.hpp"

#include <gsl/gsl_assert>
#include <details/ie_exception.hpp>
#include <cudnn.h>

#include "converters.hpp"

namespace CUDAPlugin::Convolution::Details {

ConvolutionParamsCuDnn::ConvolutionParamsCuDnn(const Convolution::Details::ConvolutionParams& params)
    : number_of_dims_{ static_cast<int>(params.NumberOfDims()) }
    , data_type_{ convertDataType<cudnnDataType_t>(params.element_type_) }
{
    if (params.padding_before_ != params.padding_after_) {
        THROW_IE_EXCEPTION << "Asymmetric padding is not supported: "
                           << "padding_before: " << params.padding_before_
                           << ", padding_after: " << params.padding_after_;
    }

    Expects(number_of_dims_ > NON_SPATIAL_DIMS_NUMBER);
    Ensures(params.input_shape_.size() == number_of_dims_);
    Ensures(params.filter_shape_.size() == number_of_dims_);
    Ensures(params.output_shape_.size() == number_of_dims_);

    const size_t number_of_spatial_dims = NumberOfSpatialDims();
    Ensures(params.strides_.size() == number_of_spatial_dims);
    Ensures(params.dilations_.size() == number_of_spatial_dims);
    Ensures(params.padding_before_.size() == number_of_spatial_dims);

    std::copy(params.input_shape_.begin(), params.input_shape_.end(), input_shape_.begin());
    std::copy(params.filter_shape_.begin(), params.filter_shape_.end(), filter_shape_.begin());
    std::copy(params.output_shape_.begin(), params.output_shape_.end(), output_shape_.begin());

    std::copy(params.strides_.begin(), params.strides_.end(), strides_.begin());
    std::copy(params.dilations_.begin(), params.dilations_.end(), dilations_.begin());
    std::copy(params.padding_before_.begin(), params.padding_before_.end(), paddings_.begin());
}

CUDA::DnnTensorDescriptor
ConvolutionParamsCuDnn::MakeInputDescriptor() const {
    return CUDA::DnnTensorDescriptor {cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, data_type_,
                                      number_of_dims_, input_shape_.data()};
}

CUDA::DnnFilterDescriptor
ConvolutionParamsCuDnn::MakeFilterDescriptor() const {
    return CUDA::DnnFilterDescriptor {data_type_, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                                      number_of_dims_, filter_shape_.data()};
}

CUDA::DnnTensorDescriptor
ConvolutionParamsCuDnn::MakeOutputDescriptor() const {
    return CUDA::DnnTensorDescriptor {cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, data_type_,
                                      number_of_dims_, output_shape_.data()};
}

CUDA::DnnConvolutionDescriptor
ConvolutionParamsCuDnn::MakeConvolutionDescriptor(cudnnDataType_t convDataType) const {
    // According to `ngraph::op::v1::Convolution` spec, it "computes 1D, 2D or 3D convolution
    // (cross-correlation to be precise)".
    constexpr cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

    // The convolution computation will be done in the specified dataType, which can be
    // potentially different from the input/output tensors.
    const cudnnDataType_t datatype = convDataType;

    CUDA::DnnConvolutionDescriptor conv_desc { NumberOfSpatialDims(), paddings_.data(),
        strides_.data(), dilations_.data(), mode, datatype };

    // Enable computations on Tensor Core hardware which requires at least Volta GPU
    // (compute capability 7.0).
    const cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
    throwIfError(::cudnnSetConvolutionMathType(conv_desc.get(), math_type));

    return conv_desc;
}


ConvolutionDescriptorsCuDnn::ConvolutionDescriptorsCuDnn(const CUDA::Device& device,
        const ConvolutionParamsCuDnn& params)
    : tensor_element_type_{ params.ElementType() }
    , input_{ params.MakeInputDescriptor() }
    , filter_{ params.MakeFilterDescriptor() }
    , output_{ params.MakeOutputDescriptor() }
    , conv_{}
    , algo_perf_{} {
    CUDA::DnnHandle dnnHandle {};
    SelectAlgo(device, dnnHandle, params);
}

void ConvolutionDescriptorsCuDnn::SelectAlgo(const CUDA::Device& device,
                                             const CUDA::DnnHandle& dnnHandle,
                                             const ConvolutionParamsCuDnn& params) {
    switch (tensor_element_type_) {
    case CUDNN_DATA_HALF:
        if (SelectAlgoForConvDataType(device, dnnHandle, params, CUDNN_DATA_HALF))
            return;
        if (SelectAlgoForConvDataType(device, dnnHandle, params, CUDNN_DATA_FLOAT))
            return;
        break;
    default:
        if (SelectAlgoForConvDataType(device, dnnHandle, params, tensor_element_type_))
            return;
    }

    THROW_IE_EXCEPTION << "cuDNN: Unsupported convolution";
}

bool ConvolutionDescriptorsCuDnn::SelectAlgoForConvDataType(const CUDA::Device& device,
                                                            const CUDA::DnnHandle& dnnHandle,
                                                            const ConvolutionParamsCuDnn& params,
                                                            cudnnDataType_t convDataType) {
    cudnnStatus_t status = CUDNN_STATUS_NOT_SUPPORTED;
    conv_ = params.MakeConvolutionDescriptor(convDataType);
    const int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    if (device.props().major < 7)
    {
        status = ::cudnnFindConvolutionForwardAlgorithm(
                                    dnnHandle.get(),
                                    input_.get(),
                                    filter_.get(),
                                    conv_.get(),
                                    output_.get(),
                                    requestedAlgoCount,
                                    &returnedAlgoCount,
                                    &algo_perf_);
    } else
    {
        status = ::cudnnGetConvolutionForwardAlgorithm_v7(
                                    dnnHandle.get(),
                                    input_.get(),
                                    filter_.get(),
                                    conv_.get(),
                                    output_.get(),
                                    requestedAlgoCount,
                                    &returnedAlgoCount,
                                    &algo_perf_);
    }
    return (status == CUDNN_STATUS_SUCCESS)
        && (algo_perf_.status == CUDNN_STATUS_SUCCESS)
        && (returnedAlgoCount > 0);
}

} // namespace CUDAPlugin
