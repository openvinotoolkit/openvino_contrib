// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_operation_base.hpp"
#include "kernels/scatter_elements_update.hpp"

namespace ov {
namespace nvidia_gpu {

class ScatterElementsUpdateOp : public OperationBase {
public:
    ScatterElementsUpdateOp(const CreationContext& context,
                            const ov::Node& node,
                            IndexCollection&& inputIds,
                            IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;

private:
    kernel::ScatterElementsUpdateParams params_{};
    size_t element_size_{};
    size_t index_size_{};
};

}  // namespace nvidia_gpu
}  // namespace ov
