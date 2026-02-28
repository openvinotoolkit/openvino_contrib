// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <openvino/op/clamp.hpp>

#include "kernels/clamp.hpp"

namespace ov {
namespace nvidia_gpu {

class ClampCudaOp : public OperationBase {
public:
    using NodeOp = ov::op::v0::Clamp;

    ClampCudaOp(const CreationContext& context,
                const NodeOp& node,
                IndexCollection&& inputIds,
                IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;

private:
    std::optional<kernel::Clamp> kernel_;
};

}  // namespace nvidia_gpu
}  // namespace ov
