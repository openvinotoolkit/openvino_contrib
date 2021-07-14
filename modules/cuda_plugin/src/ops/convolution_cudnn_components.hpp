// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/creation_context.hpp>

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
    ConvolutionDescriptorsCuDnn(const CUDA::CreationContext& context,
            const Convolution::Details::ConvolutionParamsCuDnn& params);

    cudnnDataType_t ElementType() const { return tensor_element_type_; }
    const CUDA::DnnTensorDescriptor& Input() const { return input_; }
    const CUDA::DnnFilterDescriptor& Filter() const { return filter_; }
    const CUDA::DnnTensorDescriptor& Output() const { return output_; }
    const CUDA::DnnConvolutionDescriptor& Conv() const { return conv_; }
    const cudnnConvolutionFwdAlgoPerf_t& Algo() const { return algo_perf_; }
    void FindAlgo(const CUDA::DnnHandle& dnnHandle,
                  InferenceEngine::gpu::DevicePointer<const void*> inPtr,
                  InferenceEngine::gpu::DevicePointer<const void*> filterPtr,
                  InferenceEngine::gpu::DevicePointer<void*> outPtr,
                  InferenceEngine::gpu::DeviceBuffer<uint8_t> workspace);

private:
    bool FindAlgoForConvDataType(const CUDA::DnnHandle& dnnHandle,
                                 InferenceEngine::gpu::DevicePointer<const void*> inPtr,
                                 InferenceEngine::gpu::DevicePointer<const void*> filterPtr,
                                 InferenceEngine::gpu::DevicePointer<void*> outPtr,
                                 InferenceEngine::gpu::DeviceBuffer<uint8_t> workspace,
                                 cudnnDataType_t convDataType);
    void BenchmarkOptimalAlgo(const CUDA::DnnHandle& dnnHandle,
                              const ConvolutionParamsCuDnn& params);
    void GetAlgo(const CUDA::DnnHandle& dnnHandle);
    bool GetAlgoForConvDataType(const CUDA::DnnHandle& dnnHandle,
                                cudnnDataType_t convDataType);
    void FindAlgo(const CUDA::DnnHandle& dnnHandle);
    bool FindAlgoForConvDataType(const CUDA::DnnHandle& dnnHandle,
                                 cudnnDataType_t convDataType);
private:
    CUDA::CreationContext context_;
    ConvolutionParamsCuDnn params_;
    cudnnDataType_t tensor_element_type_;
    CUDA::DnnTensorDescriptor input_;
    CUDA::DnnFilterDescriptor filter_;
    CUDA::DnnTensorDescriptor output_;
    CUDA::DnnConvolutionDescriptor conv_;
    cudnnConvolutionFwdAlgoPerf_t algo_perf_;
};


} // namespace CUDAPlugin::Convolution::Details
