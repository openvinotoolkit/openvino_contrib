// Copyright (C) 2021-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "cuda_operation_base.hpp"
#include "kernels/scatter_update.hpp"

namespace ov {
namespace nvidia_gpu {

class ScatterUpdateOp : public OperationBase {
public:
    ScatterUpdateOp(const CreationContext& context,
                    const ov::Node& node,
                    IndexCollection&& inputIds,
                    IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;

private:
    std::optional<kernel::ScatterUpdate> kernel_;
};

}  // namespace nvidia_gpu
}  // namespace ov
