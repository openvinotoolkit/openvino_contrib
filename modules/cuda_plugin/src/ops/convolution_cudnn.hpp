// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>

#include "cuda/dnn.hpp"
#include "cuda_operation_base.hpp"

namespace CUDAPlugin {

/**
 * @brief Implements `ngraph::op::v1::Convolution` using cuDNN API
 * which doesn't support asymmetric padding.
 */
class ConvolutionCuDnn : public IOperationExec {
public:
    ConvolutionCuDnn(ngraph::element::Type_t element_type,
                     const ngraph::Shape& input_shape,
                     const ngraph::Shape& filter_shape,
                     const ngraph::Shape& output_shape,
                     const ngraph::Strides& strides,
                     const ngraph::Strides& dilations,
                     const ngraph::CoordinateDiff& padding_before,
                     const ngraph::CoordinateDiff& padding_after);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) override;
    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override {}
    WorkbufferRequest GetWorkBufferRequest() const override;
    const WorkbufferIndices&  GetWorkbufferIds() const { return workbuffer_ids_; }
    WorkbufferStatus SetWorkbufferIds(WorkbufferIndices&& workbufferIds) override {
      workbuffer_ids_ = workbufferIds;
      return workbuffer_ids_.immutableIndices.empty() ? WorkbufferStatus::NoInitNeeded : WorkbufferStatus::InitNeeded;
    }

private:
    struct ConvParams {
        int arrayLength;
        std::array<int, CUDNN_DIM_MAX> padA;
        std::array<int, CUDNN_DIM_MAX> filterStrideA;
        std::array<int, CUDNN_DIM_MAX> dilationA;
    };
    void SelectAlgo(const CUDA::DnnHandle& dnnHandle,
                    const ConvParams& conv_params,
                    cudnnDataType_t tensor_element_type);
    bool SelectAlgoForConvDataType(const CUDA::DnnHandle& dnnHandle,
                                   const ConvParams& conv_params,
                                   cudnnDataType_t convDataType);

    static CUDA::DnnTensorDescriptor MakeTensorDescriptor(
        ngraph::element::Type_t element_type, const ngraph::Shape& shape);
    static CUDA::DnnFilterDescriptor MakeFilterDescriptor(
        ngraph::element::Type_t element_type, const ngraph::Shape& shape);
    static CUDA::DnnConvolutionDescriptor
        MakeConvolutionDescriptor(const ConvParams& conv_params, cudnnDataType_t convDataType);

private:
    WorkbufferIndices workbuffer_ids_;
    CUDA::DnnTensorDescriptor input_desc_;
    CUDA::DnnTensorDescriptor output_desc_;
    CUDA::DnnFilterDescriptor filter_desc_;
    const cudnnDataType_t tensor_element_type_;
    CUDA::DnnConvolutionDescriptor conv_desc_;
    cudnnConvolutionFwdAlgoPerf_t algo_perf_;
};

} // namespace CUDAPlugin
