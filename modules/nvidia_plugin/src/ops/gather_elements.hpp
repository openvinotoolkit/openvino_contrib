// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_operation_base.hpp"
#include "kernels/gather_elements.hpp"

namespace ov {
namespace nvidia_gpu {

class GatherElementsOp : public OperationBase {
public:
    GatherElementsOp(const CreationContext& context,
                     const ov::Node& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;

private:
    kernel::GatherElementsParams params_{};
    size_t element_size_{};
    size_t index_size_{};
};

}  // namespace nvidia_gpu
}  // namespace ov
