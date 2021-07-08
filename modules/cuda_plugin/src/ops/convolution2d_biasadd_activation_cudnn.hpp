// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>

#include "cuda/dnn.hpp"
#include "cuda_operation_base.hpp"
#include "convolution_cudnn_components.hpp"

namespace CUDAPlugin {

class Convolution2DBiasAddActivationCuDnn : public IOperationExec {
public:
    Convolution2DBiasAddActivationCuDnn(
        const Convolution::Details::ConvolutionBiasAddActivationParams& params);

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
    static CUDA::DnnTensorDescriptor MakeBiasDescriptor(const ngraph::Shape& shape,
                                                        ngraph::element::Type_t element_type);
    static CUDA::DnnActivationDescriptor MakeActivationDescriptor(nodes::ActivationMode mode);

private:
    WorkbufferIndices workbuffer_ids_;
    Convolution::Details::ConvolutionDescriptorsCuDnn conv_descs_;
    CUDA::DnnTensorDescriptor bias_desc_;
    CUDA::DnnActivationDescriptor activation_desc_;
};

} // namespace CUDAPlugin
