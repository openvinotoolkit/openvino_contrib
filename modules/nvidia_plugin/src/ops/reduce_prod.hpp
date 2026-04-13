// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reduce.hpp"

namespace ov {
namespace nvidia_gpu {

class ReduceProdOp : public ReduceOp {
public:
    explicit ReduceProdOp(const CreationContext& context,
                          const ov::Node& node,
                          IndexCollection&& inputIds,
                          IndexCollection&& outputIds);
};

/// Custom integer ReduceProd implementation (cuDNN does not support integer reduce).
class ReduceProdIntOp : public OperationBase {
public:
    ReduceProdIntOp(const CreationContext& context,
                    const ov::Node& node,
                    IndexCollection&& inputIds,
                    IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override {
        return CudaGraphCompatibility::FULL;
    }

private:
    size_t num_elements_;
};

}  // namespace nvidia_gpu
}  // namespace ov
