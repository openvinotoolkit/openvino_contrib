// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <openvino/op/swish.hpp>

#include "kernels/swish.hpp"

namespace ov {
namespace nvidia_gpu {

class SwishOp : public OperationBase {
public:
    SwishOp(const CreationContext& context,
            const ov::Node& node,
            IndexCollection&& inputIds,
            IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;

private:
    std::optional<kernel::Swish> kernel_;
};

}  // namespace nvidia_gpu
}  // namespace ov
