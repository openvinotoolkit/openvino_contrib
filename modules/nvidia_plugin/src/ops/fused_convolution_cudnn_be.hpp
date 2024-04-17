// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <vector>

#include "convolution_components/convolution_components.hpp"
#include "cuda/dnn_be.hpp"
#include "cuda_operation_base.hpp"
#include "ops/convolution_components/convolution_cudnn_components.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Implements `ov::op::v1::Convolution` using cuDNN Backend API.
 *
 * cuDNN Backend API was introduced in cuDNN version 8 and among other
 * features provides support for asymmetric padding.
 */
class FusedConvolutionCuDnnBE : public OperationCuDnn {
public:
    FusedConvolutionCuDnnBE(const CreationContext& context,
                            const ov::Node& node,
                            IndexCollection&& inputIds,
                            IndexCollection&& outputIds,
                            const Convolution::Details::FusedConvolutionParams& params);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    static std::shared_ptr<CUDA::DnnBETensorDescriptor> MakeTensorDescriptor(int64_t id,
                                                                             cudnnDataType_t element_type,
                                                                             const ov::Shape& shape,
                                                                             const cudnnTensorFormat_t format,
                                                                             bool isVirtual = false);
    std::shared_ptr<CUDA::DnnBEExecutionPlan> performBenchmarks(
        const CUDA::DnnHandle& context, std::vector<std::shared_ptr<CUDA::DnnBEExecutionPlan>>& plans);

    std::shared_ptr<CUDA::DnnBEEngineConfigDescriptor> engine_config_;
    int64_t workspace_size_ = 0;
    const Convolution::Details::FusedConvolutionParams params_;
};

}  // namespace nvidia_gpu
}  // namespace ov
