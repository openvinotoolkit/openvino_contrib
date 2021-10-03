// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <vector>
#include <atomic>

#include "cuda/dnn_be.hpp"
#include "cuda_operation_base.hpp"
#include "convolution_components.hpp"

namespace CUDAPlugin {

/**
 * @brief Implements `ngraph::op::v1::Convolution` using cuDNN Backend API.
 *
 * cuDNN Backend API was introduced in cuDNN version 8 and among other
 * features provides support for asymmetric padding.
 */
class ConvolutionCuDnnBE : public OperationCuDnn {
public:
    ConvolutionCuDnnBE(const CreationContext& context,
                       const ngraph::Node& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds,
                       const Convolution::Details::ConvolutionParams& params);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    bool TryExecutePlan(const InferenceRequestContext& context,
                        Inputs inputs,
                        Outputs outputs,
                        void* workbuffer,
                        const CUDA::DnnBEExecutionPlanDescriptor& plan) const;

    static CUDA::DnnBETensorDescriptor
        MakeTensorDescriptor(int64_t id, cudnnDataType_t element_type,
                             const ngraph::Shape& shape);

private:
    mutable std::atomic<int64_t> exec_plan_index_hint_ = 0;
    std::vector<CUDA::DnnBEExecutionPlanDescriptor> exec_plans_;
};

} // namespace CUDAPlugin
