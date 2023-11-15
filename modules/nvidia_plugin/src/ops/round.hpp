// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "openvino/op/round.hpp"

namespace ov {
namespace nvidia_gpu {

class RoundOp : public OperationBase {
public:
    using NodeOp = ov::op::v5::Round;

    RoundOp(const CreationContext& context,
            const NodeOp& node,
            IndexCollection&& inputIds,
            IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;

private:
    std::optional<kernel::Round> kernel_;
};

}  // namespace nvidia_gpu
}  // namespace ov
