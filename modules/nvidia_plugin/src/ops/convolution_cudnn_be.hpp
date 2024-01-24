// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <vector>

#include "convolution_components/convolution_components.hpp"
#include "cuda/dnn_be.hpp"
#include "cuda_operation_base.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Implements `ov::op::v1::Convolution` using cuDNN Backend API.
 *
 * cuDNN Backend API was introduced in cuDNN version 8 and among other
 * features provides support for asymmetric padding.
 */
class ConvolutionCuDnnBE : public OperationCuDnn {
public:
    ConvolutionCuDnnBE(const CreationContext& context,
                       const ov::Node& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds,
                       const Convolution::Details::ConvolutionParams& params);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;

    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    std::shared_ptr<CUDA::DnnBEExecutionPlan> performBenchmarks(
        const CUDA::DnnHandle& dnnHandle, std::vector<std::shared_ptr<CUDA::DnnBEExecutionPlan>>& plans);

    static std::shared_ptr<CUDA::DnnBETensorDescriptor> MakeTensorDescriptor(int64_t id,
                                                                             cudnnDataType_t element_type,
                                                                             const ov::Shape& shape);

private:
    const Convolution::Details::ConvolutionParams params_;
    std::shared_ptr<CUDA::DnnBEEngineConfigDescriptor> engine_config_;
    int64_t workspace_size_ = 0;
};

}  // namespace nvidia_gpu
}  // namespace ov
