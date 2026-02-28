// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cuda_operation_base.hpp>
#include <kernels/logical_not.cuh>

namespace ov {
namespace nvidia_gpu {

class LogicalNotOp : public OperationBase {
public:
    explicit LogicalNotOp(const CreationContext& context,
                          const std::shared_ptr<ov::Node>& node,
                          IndexCollection&& inputIds,
                          IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;

private:
    kernel::LogicalNot kernel_;
};

}  // namespace nvidia_gpu
}  // namespace ov
