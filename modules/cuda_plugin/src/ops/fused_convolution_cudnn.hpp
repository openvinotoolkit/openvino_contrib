// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>

#include "convolution_components/convolution_cudnn_components.hpp"
#include "cuda/dnn.hpp"
#include "cuda_operation_base.hpp"

namespace CUDAPlugin {

class FusedConvolutionCuDnn : public OperationCuDnn {
public:
    FusedConvolutionCuDnn(const CreationContext& context,
                          const ngraph::Node& node,
                          IndexCollection&& inputIds,
                          IndexCollection&& outputIds,
                          Convolution::Details::FusedConvolutionParams params);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override;
    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override {}
    WorkbufferRequest GetWorkBufferRequest() const override;
    const WorkbufferIds& GetWorkbufferIds() const override { return workbuffer_ids_; }
    WorkbufferStatus SetWorkbufferIds(WorkbufferIds&& workbufferIds) override {
        workbuffer_ids_ = workbufferIds;
        return workbuffer_ids_.immutableIds.empty() ? WorkbufferStatus::NoInitNeeded : WorkbufferStatus::InitNeeded;
    }

private:
    static CUDA::DnnTensorDescriptor MakeBiasDescriptor(const ngraph::Shape& shape,
                                                        ngraph::element::Type_t element_type);
    static CUDA::DnnActivationDescriptor MakeActivationDescriptor(nodes::ActivationMode mode);

private:
    WorkbufferIds workbuffer_ids_;
    Convolution::Details::ConvolutionDescriptorsCuDnn conv_descs_;
    CUDA::DnnTensorDescriptor bias_desc_;
    std::optional<CUDA::DnnTensorDescriptor> add_desc_;
    CUDA::DnnActivationDescriptor activation_desc_;
};

}  // namespace CUDAPlugin
