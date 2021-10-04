// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_cudnn_components.hpp"

#include <cudnn.h>
#include <fmt/ostream.h>

#include <cuda/cuda_config.hpp>
#include <cuda_config.hpp>
#include <gsl/gsl_assert>
#include <ngraph/util.hpp>

#include "converters.hpp"

namespace CUDAPlugin::Convolution::Details {

ConvolutionParamsCuDnn::ConvolutionParamsCuDnn(const Convolution::Details::ConvolutionParams& params)
    : number_of_dims_{static_cast<int>(params.NumberOfDims())},
      data_type_{convertDataType<cudnnDataType_t>(params.element_type_)} {
    if (params.padding_before_ != params.padding_after_) {
        throwIEException(
            fmt::format("Asymmetric padding is not supported: padding_before: "
                        "{}, padding_after: {}",
                        params.padding_before_,
                        params.padding_after_));
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

CUDA::DnnTensorDescriptor ConvolutionParamsCuDnn::MakeInputDescriptor() const {
    return CUDA::DnnTensorDescriptor{}.set(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, data_type_, number_of_dims_, input_shape_.data());
}

CUDA::DnnFilterDescriptor ConvolutionParamsCuDnn::MakeFilterDescriptor() const {
    return CUDA::DnnFilterDescriptor{}.set(data_type_, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, number_of_dims_, filter_shape_.data());
}

CUDA::DnnTensorDescriptor ConvolutionParamsCuDnn::MakeOutputDescriptor() const {
    return CUDA::DnnTensorDescriptor{}.set(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, data_type_, number_of_dims_, output_shape_.data());
}

CUDA::DnnConvolutionDescriptor ConvolutionParamsCuDnn::MakeConvolutionDescriptor(cudnnDataType_t convDataType) const {
    // According to `ngraph::op::v1::Convolution` spec, it "computes 1D, 2D or 3D convolution
    // (cross-correlation to be precise)".
    constexpr cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

    // The convolution computation will be done in the specified dataType, which can be
    // potentially different from the input/output tensors.
    const cudnnDataType_t datatype = convDataType;

    CUDA::DnnConvolutionDescriptor conv_desc;
    conv_desc.set(NumberOfSpatialDims(), paddings_.data(), strides_.data(), dilations_.data(), mode, datatype);

    // Enable computations on Tensor Core hardware which requires at least Volta GPU
    // (compute capability 7.0).
    const cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
    throwIfError(::cudnnSetConvolutionMathType(conv_desc.get(), math_type));

    return conv_desc;
}

ConvolutionDescriptorsCuDnn::ConvolutionDescriptorsCuDnn(const CreationContext& context,
                                                         const ConvolutionParamsCuDnn& params)
    : params_{params},
      tensor_element_type_{params_.ElementType()},
      input_{params_.MakeInputDescriptor()},
      filter_{params_.MakeFilterDescriptor()},
      output_{params_.MakeOutputDescriptor()},
      conv_{},
      algo_perf_{} {
    auto& dnnHandle = context.dnnHandle();
    if (context.optimizeOption()) {
        BenchmarkOptimalAlgo(dnnHandle, params_);
    } else {
        GetAlgo(dnnHandle);
    }
}

void ConvolutionDescriptorsCuDnn::BenchmarkOptimalAlgo(const CUDA::DnnHandle& dnnHandle,
                                                       const ConvolutionParamsCuDnn& params) {
    constexpr auto kNumSelectAlgo = 3;
    int convForwardAlgorithmMaxCount;
    throwIfError(cudnnGetConvolutionForwardAlgorithmMaxCount(dnnHandle.get(), &convForwardAlgorithmMaxCount));
    std::vector<int> timesCuDNNAlgosSelected(convForwardAlgorithmMaxCount);
    std::array<cudnnConvolutionFwdAlgoPerf_t, kNumSelectAlgo> cudnnAlgos{};
    for (auto& algo : cudnnAlgos) {
        FindAlgo(dnnHandle);
        algo = algo_perf_;
        IE_ASSERT(algo_perf_.algo >= 0);
        IE_ASSERT(algo_perf_.algo < convForwardAlgorithmMaxCount);
        timesCuDNNAlgosSelected[algo_perf_.algo] += 1;
    }
    auto maxAlgoIter = std::max_element(timesCuDNNAlgosSelected.begin(), timesCuDNNAlgosSelected.end());
    auto optimalAlgoId =
        static_cast<cudnnConvolutionFwdAlgo_t>(std::distance(timesCuDNNAlgosSelected.begin(), maxAlgoIter));
    auto optimalAlgo = std::find_if(
        cudnnAlgos.begin(), cudnnAlgos.end(), [optimalAlgoId](const auto& a) { return a.algo == optimalAlgoId; });
    algo_perf_ = *optimalAlgo;
}

void ConvolutionDescriptorsCuDnn::GetAlgo(const CUDA::DnnHandle& dnnHandle) {
    switch (tensor_element_type_) {
        case CUDNN_DATA_HALF:
            if (GetAlgoForConvDataType(dnnHandle, CUDNN_DATA_HALF)) return;
            if (GetAlgoForConvDataType(dnnHandle, CUDNN_DATA_FLOAT)) return;
            break;
        default:
            if (GetAlgoForConvDataType(dnnHandle, tensor_element_type_)) return;
    }

    throwIEException("cuDNN: Unsupported convolution");
}

bool ConvolutionDescriptorsCuDnn::GetAlgoForConvDataType(const CUDA::DnnHandle& dnnHandle,
                                                         cudnnDataType_t convDataType) {
    cudnnStatus_t status = CUDNN_STATUS_NOT_SUPPORTED;
    conv_ = params_.MakeConvolutionDescriptor(convDataType);
    const int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    status = ::cudnnGetConvolutionForwardAlgorithm_v7(dnnHandle.get(),
                                                      input_.get(),
                                                      filter_.get(),
                                                      conv_.get(),
                                                      output_.get(),
                                                      requestedAlgoCount,
                                                      &returnedAlgoCount,
                                                      &algo_perf_);

    if ((status != CUDNN_STATUS_SUCCESS) || (algo_perf_.status != CUDNN_STATUS_SUCCESS) || (returnedAlgoCount <= 0)) {
        return false;
    }

    size_t sizeInBytes = 0;
    throwIfError(::cudnnGetConvolutionForwardWorkspaceSize(
        dnnHandle.get(), input_.get(), filter_.get(), conv_.get(), output_.get(), algo_perf_.algo, &sizeInBytes));
    algo_perf_.memory = sizeInBytes;

    return true;
}

void ConvolutionDescriptorsCuDnn::FindAlgo(const CUDA::DnnHandle& dnnHandle) {
    switch (tensor_element_type_) {
        case CUDNN_DATA_HALF:
            if (FindAlgoForConvDataType(dnnHandle, CUDNN_DATA_HALF)) return;
            if (FindAlgoForConvDataType(dnnHandle, CUDNN_DATA_FLOAT)) return;
            break;
        default:
            if (FindAlgoForConvDataType(dnnHandle, tensor_element_type_)) return;
    }

    throwIEException("cuDNN: Unsupported convolution");
}

bool ConvolutionDescriptorsCuDnn::FindAlgoForConvDataType(const CUDA::DnnHandle& dnnHandle,
                                                          cudnnDataType_t convDataType) {
    cudnnStatus_t status = CUDNN_STATUS_NOT_SUPPORTED;
    conv_ = params_.MakeConvolutionDescriptor(convDataType);
    const int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    status = ::cudnnFindConvolutionForwardAlgorithm(dnnHandle.get(),
                                                    input_.get(),
                                                    filter_.get(),
                                                    conv_.get(),
                                                    output_.get(),
                                                    requestedAlgoCount,
                                                    &returnedAlgoCount,
                                                    &algo_perf_);

    if ((status != CUDNN_STATUS_SUCCESS) || (algo_perf_.status != CUDNN_STATUS_SUCCESS) || (returnedAlgoCount <= 0)) {
        return false;
    }

    size_t sizeInBytes = 0;
    throwIfError(::cudnnGetConvolutionForwardWorkspaceSize(
        dnnHandle.get(), input_.get(), filter_.get(), conv_.get(), output_.get(), algo_perf_.algo, &sizeInBytes));
    algo_perf_.memory = sizeInBytes;

    return true;
}

void ConvolutionDescriptorsCuDnn::FindAlgo(const CUDA::DnnHandle& dnnHandle,
                                           CUDA::DevicePointer<const void*> inPtr,
                                           CUDA::DevicePointer<const void*> filterPtr,
                                           CUDA::DevicePointer<void*> outPtr,
                                           CUDA::DeviceBuffer<std::byte> workspace) {
    switch (tensor_element_type_) {
        case CUDNN_DATA_HALF:
            if (FindAlgoForConvDataType(dnnHandle, inPtr, filterPtr, outPtr, workspace, CUDNN_DATA_HALF)) return;
            if (FindAlgoForConvDataType(dnnHandle, inPtr, filterPtr, outPtr, workspace, CUDNN_DATA_FLOAT)) return;
            break;
        default:
            if (FindAlgoForConvDataType(dnnHandle, inPtr, filterPtr, outPtr, workspace, tensor_element_type_)) return;
    }

    throwIEException("cuDNN: Unsupported convolution");
}

bool ConvolutionDescriptorsCuDnn::FindAlgoForConvDataType(const CUDA::DnnHandle& dnnHandle,
                                                          CUDA::DevicePointer<const void*> inPtr,
                                                          CUDA::DevicePointer<const void*> filterPtr,
                                                          CUDA::DevicePointer<void*> outPtr,
                                                          CUDA::DeviceBuffer<std::byte> workspace,
                                                          cudnnDataType_t convDataType) {
    cudnnStatus_t status = CUDNN_STATUS_NOT_SUPPORTED;
    conv_ = params_.MakeConvolutionDescriptor(convDataType);
    const int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    status = ::cudnnFindConvolutionForwardAlgorithmEx(dnnHandle.get(),
                                                      input_.get(),
                                                      inPtr.get(),
                                                      filter_.get(),
                                                      filterPtr.get(),
                                                      conv_.get(),
                                                      output_.get(),
                                                      outPtr.get(),
                                                      requestedAlgoCount,
                                                      &returnedAlgoCount,
                                                      &algo_perf_,
                                                      workspace.data(),
                                                      workspace.size());
    return (status == CUDNN_STATUS_SUCCESS) && (algo_perf_.status == CUDNN_STATUS_SUCCESS) && (returnedAlgoCount > 0);
}

ConvolutionBackpropDataParamsCuDnn::ConvolutionBackpropDataParamsCuDnn(
    const Convolution::Details::ConvolutionBackwardDataParams& params)
    : number_of_dims_{static_cast<int>(params.NumberOfDims())},
      data_type_{convertDataType<cudnnDataType_t>(params.element_type_)} {
    if (params.pads_begin_ != params.pads_end_) {
        throwIEException(
            fmt::format("Asymmetric padding is not supported: padding_before: "
                        "{}, padding_after: {}",
                        params.pads_begin_,
                        params.pads_end_));
    }

    Expects(number_of_dims_ > NON_SPATIAL_DIMS_NUMBER);
    Ensures(params.doutput_shape_.size() == number_of_dims_);
    Ensures(params.filter_shape_.size() == number_of_dims_);
    Ensures(params.dinput_shape_.size() == number_of_dims_);

    const size_t number_of_spatial_dims = NumberOfSpatialDims();
    Ensures(params.strides_.size() == number_of_spatial_dims);
    Ensures(params.dilations_.size() == number_of_spatial_dims);
    Ensures(params.pads_begin_.size() == number_of_spatial_dims);

    std::copy(params.doutput_shape_.begin(), params.doutput_shape_.end(), doutput_shape_.begin());
    std::copy(params.filter_shape_.begin(), params.filter_shape_.end(), filter_shape_.begin());
    std::copy(params.dinput_shape_.begin(), params.dinput_shape_.end(), dinput_shape_.begin());

    std::copy(params.strides_.begin(), params.strides_.end(), strides_.begin());
    std::copy(params.dilations_.begin(), params.dilations_.end(), dilations_.begin());
    std::copy(params.pads_begin_.begin(), params.pads_begin_.end(), paddings_.begin());
}

CUDA::DnnTensorDescriptor ConvolutionBackpropDataParamsCuDnn::MakeDOutputDescriptor() const {
    return CUDA::DnnTensorDescriptor{}.set(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, data_type_, number_of_dims_, doutput_shape_.data());
}

CUDA::DnnFilterDescriptor ConvolutionBackpropDataParamsCuDnn::MakeFilterDescriptor() const {
    return CUDA::DnnFilterDescriptor{}.set(data_type_, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, number_of_dims_, filter_shape_.data());
}

CUDA::DnnTensorDescriptor ConvolutionBackpropDataParamsCuDnn::MakeDInputDescriptor() const {
    return CUDA::DnnTensorDescriptor{}.set(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, data_type_, number_of_dims_, dinput_shape_.data());
}

CUDA::DnnConvolutionDescriptor ConvolutionBackpropDataParamsCuDnn::MakeConvolutionDescriptor(
    cudnnDataType_t convDataType) const {
    // According to `ngraph::op::v1::Convolution` spec, it "computes 1D, 2D or 3D convolution
    // (cross-correlation to be precise)".
    constexpr cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

    // The convolution computation will be done in the specified dataType, which can be
    // potentially different from the input/output tensors.
    const cudnnDataType_t datatype = convDataType;

    CUDA::DnnConvolutionDescriptor conv_desc;
    conv_desc.set(NumberOfSpatialDims(), paddings_.data(), strides_.data(), dilations_.data(), mode, datatype);

    // Enable computations on Tensor Core hardware which requires at least Volta GPU
    // (compute capability 7.0).
    const cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
    throwIfError(::cudnnSetConvolutionMathType(conv_desc.get(), math_type));

    return conv_desc;
}

ConvolutionBackpropDataDescriptorCuDnn::ConvolutionBackpropDataDescriptorCuDnn(
    const CreationContext& context, const ConvolutionBackpropDataParamsCuDnn& params)
    : params_{params},
      tensor_element_type_{params_.ElementType()},
      filter_desc_{params_.MakeFilterDescriptor()},
      doutput_desc_{params_.MakeDOutputDescriptor()},
      dinput_desc_{params_.MakeDInputDescriptor()},
      conv_{},
      algo_perf_{} {
    auto& dnnHandle = context.dnnHandle();
    if (context.optimizeOption()) {
        BenchmarkOptimalAlgo(dnnHandle);
    } else {
        GetAlgo(dnnHandle);
    }
}

void ConvolutionBackpropDataDescriptorCuDnn::BenchmarkOptimalAlgo(const CUDA::DnnHandle& dnnHandle) {
    constexpr auto kNumSelectAlgo = 3;
    int convBackwardDataAlgorithmMaxCount;
    throwIfError(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(dnnHandle.get(), &convBackwardDataAlgorithmMaxCount));
    std::vector<int> timesCuDNNAlgosSelected(convBackwardDataAlgorithmMaxCount);
    std::array<cudnnConvolutionBwdDataAlgoPerf_t, kNumSelectAlgo> cudnnAlgos{};
    for (auto& algo : cudnnAlgos) {
        FindAlgo(dnnHandle);
        algo = algo_perf_;
        IE_ASSERT(algo_perf_.algo >= 0);
        IE_ASSERT(algo_perf_.algo < convBackwardDataAlgorithmMaxCount);
        timesCuDNNAlgosSelected[algo_perf_.algo] += 1;
    }
    auto maxAlgoIter = std::max_element(timesCuDNNAlgosSelected.begin(), timesCuDNNAlgosSelected.end());
    auto optimalAlgoId =
        static_cast<cudnnConvolutionBwdDataAlgo_t>(std::distance(timesCuDNNAlgosSelected.begin(), maxAlgoIter));
    auto optimalAlgo = std::find_if(
        cudnnAlgos.begin(), cudnnAlgos.end(), [optimalAlgoId](const auto& a) { return a.algo == optimalAlgoId; });
    algo_perf_ = *optimalAlgo;
}

void ConvolutionBackpropDataDescriptorCuDnn::GetAlgo(const CUDA::DnnHandle& dnnHandle) {
    switch (tensor_element_type_) {
        case CUDNN_DATA_HALF:
            if (GetAlgoForConvDataType(dnnHandle, CUDNN_DATA_HALF)) return;
            if (GetAlgoForConvDataType(dnnHandle, CUDNN_DATA_FLOAT)) return;
            break;
        default:
            if (GetAlgoForConvDataType(dnnHandle, tensor_element_type_)) return;
    }

    throwIEException("cuDNN: Unsupported convolution");
}

bool ConvolutionBackpropDataDescriptorCuDnn::GetAlgoForConvDataType(const CUDA::DnnHandle& dnnHandle,
                                                                    cudnnDataType_t convDataType) {
    cudnnStatus_t status = CUDNN_STATUS_NOT_SUPPORTED;
    conv_ = params_.MakeConvolutionDescriptor(convDataType);
    const int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    status = ::cudnnGetConvolutionBackwardDataAlgorithm_v7(dnnHandle.get(),
                                                           filter_desc_.get(),
                                                           doutput_desc_.get(),
                                                           conv_.get(),
                                                           dinput_desc_.get(),
                                                           requestedAlgoCount,
                                                           &returnedAlgoCount,
                                                           &algo_perf_);

    if ((status != CUDNN_STATUS_SUCCESS) || (algo_perf_.status != CUDNN_STATUS_SUCCESS) || (returnedAlgoCount <= 0)) {
        return false;
    }

    size_t sizeInBytes = 0;
    throwIfError(::cudnnGetConvolutionBackwardDataWorkspaceSize(dnnHandle.get(),
                                                                filter_desc_.get(),
                                                                doutput_desc_.get(),
                                                                conv_.get(),
                                                                dinput_desc_.get(),
                                                                algo_perf_.algo,
                                                                &sizeInBytes));
    algo_perf_.memory = sizeInBytes;

    return true;
}

void ConvolutionBackpropDataDescriptorCuDnn::FindAlgo(const CUDA::DnnHandle& dnnHandle) {
    switch (tensor_element_type_) {
        case CUDNN_DATA_HALF:
            if (FindAlgoForConvDataType(dnnHandle, CUDNN_DATA_HALF)) return;
            if (FindAlgoForConvDataType(dnnHandle, CUDNN_DATA_FLOAT)) return;
            break;
        default:
            if (FindAlgoForConvDataType(dnnHandle, tensor_element_type_)) return;
    }

    throwIEException("cuDNN: Unsupported convolution");
}

bool ConvolutionBackpropDataDescriptorCuDnn::FindAlgoForConvDataType(const CUDA::DnnHandle& dnnHandle,
                                                                     cudnnDataType_t convDataType) {
    cudnnStatus_t status = CUDNN_STATUS_NOT_SUPPORTED;
    conv_ = params_.MakeConvolutionDescriptor(convDataType);
    const int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    status = ::cudnnFindConvolutionBackwardDataAlgorithm(dnnHandle.get(),
                                                         filter_desc_.get(),
                                                         doutput_desc_.get(),
                                                         conv_.get(),
                                                         dinput_desc_.get(),
                                                         requestedAlgoCount,
                                                         &returnedAlgoCount,
                                                         &algo_perf_);

    if ((status != CUDNN_STATUS_SUCCESS) || (algo_perf_.status != CUDNN_STATUS_SUCCESS) || (returnedAlgoCount <= 0)) {
        return false;
    }

    size_t sizeInBytes = 0;
    throwIfError(::cudnnGetConvolutionBackwardDataWorkspaceSize(dnnHandle.get(),
                                                                filter_desc_.get(),
                                                                doutput_desc_.get(),
                                                                conv_.get(),
                                                                dinput_desc_.get(),
                                                                algo_perf_.algo,
                                                                &sizeInBytes));
    algo_perf_.memory = sizeInBytes;

    return true;
}

void ConvolutionBackpropDataDescriptorCuDnn::FindAlgo(const CUDA::DnnHandle& dnnHandle,
                                                      CUDA::DevicePointer<const void*> filterPtr,
                                                      CUDA::DevicePointer<const void*> dInPtr,
                                                      CUDA::DevicePointer<void*> dOutPtr,
                                                      CUDA::DeviceBuffer<std::byte> workspace) {
    switch (tensor_element_type_) {
        case CUDNN_DATA_HALF:
            if (FindAlgoForConvDataType(dnnHandle, filterPtr, dInPtr, dOutPtr, workspace, CUDNN_DATA_HALF)) return;
            if (FindAlgoForConvDataType(dnnHandle, filterPtr, dInPtr, dOutPtr, workspace, CUDNN_DATA_FLOAT)) return;
            break;
        default:
            if (FindAlgoForConvDataType(dnnHandle, filterPtr, dInPtr, dOutPtr, workspace, tensor_element_type_)) return;
    }

    throwIEException("cuDNN: Unsupported convolution");
}

bool ConvolutionBackpropDataDescriptorCuDnn::FindAlgoForConvDataType(const CUDA::DnnHandle& dnnHandle,
                                                                     CUDA::DevicePointer<const void*> filterPtr,
                                                                     CUDA::DevicePointer<const void*> dInPtr,
                                                                     CUDA::DevicePointer<void*> dOutPtr,
                                                                     CUDA::DeviceBuffer<std::byte> workspace,
                                                                     cudnnDataType_t convDataType) {
    cudnnStatus_t status = CUDNN_STATUS_NOT_SUPPORTED;
    conv_ = params_.MakeConvolutionDescriptor(convDataType);
    const int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    status = ::cudnnFindConvolutionBackwardDataAlgorithmEx(dnnHandle.get(),
                                                           filter_desc_.get(),
                                                           filterPtr.get(),
                                                           doutput_desc_.get(),
                                                           dInPtr.get(),
                                                           conv_.get(),
                                                           dinput_desc_.get(),
                                                           dOutPtr.get(),
                                                           requestedAlgoCount,
                                                           &returnedAlgoCount,
                                                           &algo_perf_,
                                                           workspace.data(),
                                                           workspace.size());
    return (status == CUDNN_STATUS_SUCCESS) && (algo_perf_.status == CUDNN_STATUS_SUCCESS) && (returnedAlgoCount > 0);
}

}  // namespace CUDAPlugin::Convolution::Details
