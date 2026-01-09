// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <kernels/gather.hpp>

namespace ov {
namespace nvidia_gpu {

class GatherOp : public OperationBase {
public:
    GatherOp(const CreationContext& context,
             const ov::Node& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;

private:
    std::optional<kernel::Gather> gather_kernel_;
};

}  // namespace nvidia_gpu
}  // namespace ov
