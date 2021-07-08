// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_components.hpp"
#include "cuda/dnn.hpp"

namespace CUDAPlugin::Convolution::Details {

/**
 * @brief Presents convolution parameters in a form suitable for cuDNN API.
 */
class ConvolutionParamsCuDnn {
public:
    ConvolutionParamsCuDnn(const Convolution::Details::ConvolutionParams& params);

    int NumberOfSpatialDims() const { return number_of_dims_ - NON_SPATIAL_DIMS_NUMBER; }
    const cudnnDataType_t ElementType() const { return data_type_; }

    CUDA::DnnTensorDescriptor MakeInputDescriptor() const;
    CUDA::DnnFilterDescriptor MakeFilterDescriptor() const;
    CUDA::DnnTensorDescriptor MakeOutputDescriptor() const;
    CUDA::DnnConvolutionDescriptor MakeConvolutionDescriptor(cudnnDataType_t convDataType) const;

private:
    const int number_of_dims_;
    const cudnnDataType_t data_type_;
    using IntArray = std::array<int, CUDNN_DIM_MAX>;
    IntArray input_shape_;
    IntArray filter_shape_;
    IntArray output_shape_;
    IntArray strides_;
    IntArray dilations_;
    IntArray paddings_;
};


/**
 * @brief Prepares all data required for cuDNN convolution API invocation.
 */
class ConvolutionDescriptorsCuDnn {
public:
    ConvolutionDescriptorsCuDnn(const Convolution::Details::ConvolutionParamsCuDnn& params);

    cudnnDataType_t ElementType() const { return tensor_element_type_; }
    const CUDA::DnnTensorDescriptor& Input() const { return input_; }
    const CUDA::DnnFilterDescriptor& Filter() const { return filter_; }
    const CUDA::DnnTensorDescriptor& Output() const { return output_; }
    const CUDA::DnnConvolutionDescriptor& Conv() const { return conv_; }
    const cudnnConvolutionFwdAlgoPerf_t& Algo() const { return algo_perf_; }

private:
    void SelectAlgo(const CUDA::DnnHandle& dnnHandle,
                    const Convolution::Details::ConvolutionParamsCuDnn& params);
    bool SelectAlgoForConvDataType(const CUDA::DnnHandle& dnnHandle,
                                   const Convolution::Details::ConvolutionParamsCuDnn& params,
                                   cudnnDataType_t convDataType);
private:
    cudnnDataType_t tensor_element_type_;
    CUDA::DnnTensorDescriptor input_;
    CUDA::DnnFilterDescriptor filter_;
    CUDA::DnnTensorDescriptor output_;
    CUDA::DnnConvolutionDescriptor conv_;
    cudnnConvolutionFwdAlgoPerf_t algo_perf_;
};


} // namespace CUDAPlugin::Convolution::Details
