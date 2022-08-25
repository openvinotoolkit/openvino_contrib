// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "kernels/floor.hpp"

namespace ov {
namespace nvidia_gpu {

class FloorOp : public OperationBase {
public:
    FloorOp(const CreationContext& context,
            const ov::Node& node,
            IndexCollection&& inputIds,
            IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

private:
    std::optional<kernel::Floor> kernel_;
};

}  // namespace nvidia_gpu
}  // namespace ov
