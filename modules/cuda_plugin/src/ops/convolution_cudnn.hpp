// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>

#include "cuda_operation_base.hpp"
#include "convolution_cudnn_components.hpp"

namespace CUDAPlugin {

/**
 * @brief Implements `ngraph::op::v1::Convolution` using cuDNN API
 * which doesn't support asymmetric padding.
 */
class ConvolutionCuDnn : public IOperationExec {
public:
    ConvolutionCuDnn(const CUDA::Device& device, const Convolution::Details::ConvolutionParams& params);

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
    WorkbufferIndices workbuffer_ids_;
    Convolution::Details::ConvolutionDescriptorsCuDnn descs_;
};

} // namespace CUDAPlugin
