// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gsl/gsl_assert>
#include <details/ie_exception.hpp>

#include "convolution_cudnn.hpp"
#include "convolution.hpp"
#include "converters.hpp"
#include "constant_factory.hpp"
#include <cudnn.h>

namespace CUDAPlugin {

constexpr int NOT_SPATIAL_DIMS_COUNT = 2;

ConvolutionCuDnn::ConvolutionCuDnn(ngraph::element::Type_t element_type,
                                   const ngraph::Shape& input_shape,
                                   const ngraph::Shape& filter_shape,
                                   const ngraph::Shape& output_shape,
                                   const ngraph::Strides& strides,
                                   const ngraph::Strides& dilations,
                                   const ngraph::CoordinateDiff& padding_before,
                                   const ngraph::CoordinateDiff& padding_after)
    : input_desc_{ MakeTensorDescriptor(element_type, input_shape) }
    , output_desc_{ MakeTensorDescriptor(element_type, output_shape) }
    , filter_desc_{ MakeFilterDescriptor(element_type, filter_shape) }
    , tensor_element_type_{convertDataType<cudnnDataType_t>(element_type)} {
    ConvParams conv_params {};

    // Convolution dimension according to op spec (1D, 2D or 3D). 1D should already be
    // turned into 2D at this point.
    const int arrayLength = static_cast<int>(input_shape.size()) - NOT_SPATIAL_DIMS_COUNT;
    Expects((arrayLength == 2) || (arrayLength == 3));
    Expects(arrayLength == strides.size());
    Expects(arrayLength == dilations.size());
    Expects(arrayLength == padding_before.size());
    Expects(arrayLength == padding_after.size());
    conv_params.arrayLength = arrayLength;

    // Array of dimension arrayLength containing the zero-padding size for each dimension.
    // For every dimension, the padding represents the number of extra zeros implicitly
    // concatenated at the start and at the end of every element of that dimension.
    if (padding_before == padding_after) {
        std::copy(padding_before.begin(), padding_before.end(), conv_params.padA.begin());
    } else {
        THROW_IE_EXCEPTION << "Asymmetric padding is not supported: "
                           << "padding_before: " << padding_before
                           << ", padding_after: " << padding_after;
    }

    // Array of dimension arrayLength containing the filter stride for each dimension.
    // For every dimension, the filter stride represents the number of elements to slide to
    // reach the next start of the filtering window of the next point.
    std::copy(strides.begin(), strides.end(), conv_params.filterStrideA.begin());

    // Array of dimension arrayLength containing the dilation factor for each dimension.
    std::copy(dilations.begin(), dilations.end(), conv_params.dilationA.begin());

    // Create convolution descriptor and ask cuDNN to select appropriate algorithm.
    CUDA::DnnHandle dnnHandle {};
    SelectAlgo(dnnHandle, conv_params, tensor_element_type_);
}

void ConvolutionCuDnn::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs) {
    Expects(inputs.size() == 2);
    Expects(outputs.size() == 1);
    const CUDA::Allocation workSpace = context.getThreadContext().stream().malloc(algo_perf_.memory);
    cudnnStatus_t status = ::cudnnConvolutionForward(
                                context.getThreadContext().dnnHandle().get(),
                                &DynamicConst<constants::one>(tensor_element_type_),
                                input_desc_.get(),
                                inputs[ConvolutionOp::ArgIndices::input].get(),
                                filter_desc_.get(),
                                inputs[ConvolutionOp::ArgIndices::filter].get(),
                                conv_desc_.get(),
                                algo_perf_.algo,
                                workSpace.get(),
                                algo_perf_.memory,
                                &DynamicConst<constants::zero>(tensor_element_type_),
                                output_desc_.get(),
                                outputs[ConvolutionOp::ArgIndices::output].get());
    CUDA::throwIfError(status);
}

void ConvolutionCuDnn::SelectAlgo(const CUDA::DnnHandle& dnnHandle,
                                  const ConvParams& conv_params,
                                  cudnnDataType_t tensor_element_type) {
    switch (tensor_element_type) {
    case CUDNN_DATA_HALF:
        if (SelectAlgoForConvDataType(dnnHandle, conv_params, CUDNN_DATA_HALF))
            return;
        if (SelectAlgoForConvDataType(dnnHandle, conv_params, CUDNN_DATA_FLOAT))
            return;
        break;
    default:
        if (SelectAlgoForConvDataType(dnnHandle, conv_params, tensor_element_type))
            return;
    }

    THROW_IE_EXCEPTION << "cuDNN: Unsupported convolution";
}

bool ConvolutionCuDnn::SelectAlgoForConvDataType(const CUDA::DnnHandle& dnnHandle,
                                                 const ConvParams& conv_params,
                                                 cudnnDataType_t convDataType) {
    conv_desc_ = MakeConvolutionDescriptor(conv_params, convDataType);
    const int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    cudnnStatus_t status = ::cudnnGetConvolutionForwardAlgorithm_v7(
                                dnnHandle.get(),
                                input_desc_.get(),
                                filter_desc_.get(),
                                conv_desc_.get(),
                                output_desc_.get(),
                                requestedAlgoCount,
                                &returnedAlgoCount,
                                &algo_perf_);
    return (status == CUDNN_STATUS_SUCCESS)
        && (algo_perf_.status == CUDNN_STATUS_SUCCESS)
        && (returnedAlgoCount > 0);
}

CUDA::DnnTensorDescriptor
ConvolutionCuDnn::MakeTensorDescriptor(ngraph::element::Type_t element_type,
                                       const ngraph::Shape& shape) {
    const cudnnDataType_t datatype = convertDataType<cudnnDataType_t>(element_type);
    const int nbDims = shape.size();

    if (nbDims < 4 || nbDims > 5)
        THROW_IE_EXCEPTION << "Unexpected number of dimensions for Convolution input/output: "
                           << nbDims;

    std::array<int, CUDNN_DIM_MAX> dimA {};
    std::copy(shape.begin(), shape.end(), dimA.begin());
    return CUDA::DnnTensorDescriptor {cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, datatype,
                                      nbDims, dimA.data()};
}

CUDA::DnnFilterDescriptor
ConvolutionCuDnn::MakeFilterDescriptor(ngraph::element::Type_t element_type,
                                       const ngraph::Shape& shape) {
    const cudnnDataType_t datatype = convertDataType<cudnnDataType_t>(element_type);
    const int nbDims = shape.size();

    if (nbDims < 4 || nbDims > 5)
        THROW_IE_EXCEPTION << "Unexpected number of dimensions for Convolution filter: "
                           << nbDims;

    // Array of dimension nbDims containing the size of the filter for each dimension.
    std::array<int, CUDNN_DIM_MAX> dimA {};
    std::copy(shape.begin(), shape.end(), dimA.begin());
    return CUDA::DnnFilterDescriptor {datatype, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                                      nbDims, dimA.data()};
}

CUDA::DnnConvolutionDescriptor
ConvolutionCuDnn::MakeConvolutionDescriptor(const ConvParams& conv_params,
                                            cudnnDataType_t convDataType) {
    // According to `ngraph::op::v1::Convolution` spec, it "computes 1D, 2D or 3D convolution
    // (cross-correlation to be precise)".
    constexpr cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

    // The convolution computation will be done in the specified dataType, which can be
    // potentially different from the input/output tensors.
    const cudnnDataType_t datatype = convDataType;

    CUDA::DnnConvolutionDescriptor conv_desc { conv_params.arrayLength, conv_params.padA.data(),
        conv_params.filterStrideA.data(), conv_params.dilationA.data(), mode, datatype };

    // Enable computations on Tensor Core hardware which requires at least Volta GPU
    // (compute capability 7.0).
    const cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
    CUDA::throwIfError(::cudnnSetConvolutionMathType(conv_desc.get(), math_type));

    return conv_desc;
}

} // namespace CUDAPlugin
