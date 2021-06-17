// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <vector>
#include <atomic>

#include "cuda/dnn_be.hpp"
#include "cuda_operation_base.hpp"


namespace CUDAPlugin {

/**
 * @brief Implements `ngraph::op::v1::Convolution` using cuDNN Backend API.
 *
 * cuDNN Backend API was introduced in cuDNN version 8 and among other
 * features provides support for asymmetric padding.
 */
class ConvolutionCuDnnBE : public IOperationExec {
public:
    ConvolutionCuDnnBE(ngraph::element::Type_t element_type,
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
                 const Workbuffers& workbuffers) override;
    void InitSharedImmutableWorkbuffers(const Buffers&) override {}
    WorkbufferRequest GetWorkBufferRequest() const override;
    const WorkbufferIndices&  GetWorkbufferIds() const { return workbuffer_ids_; }
    WorkbufferStatus SetWorkbufferIds(WorkbufferIndices&& workbufferIds) override {
      workbuffer_ids_ = workbufferIds;
      return workbuffer_ids_.immutableIndices.empty() ? WorkbufferStatus::NoInitNeeded : WorkbufferStatus::InitNeeded;
    }

private:
    bool TryExecutePlan(const InferenceRequestContext& context,
                        Inputs inputs, Outputs outputs,
                        void* workbuffer,
                        const CUDA::DnnBEExecutionPlanDescriptor& plan);

    static CUDA::DnnBETensorDescriptor
        MakeTensorDescriptor(int64_t id, cudnnDataType_t element_type,
                             const ngraph::Shape& shape);

private:
    WorkbufferIndices workbuffer_ids_;
    std::atomic<int64_t> exec_plan_index_hint_;
    std::vector<CUDA::DnnBEExecutionPlanDescriptor> exec_plans_;
};

} // namespace CUDAPlugin
