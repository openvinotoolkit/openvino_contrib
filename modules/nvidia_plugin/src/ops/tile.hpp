// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_operation_base.hpp"
#include "kernels/tile.hpp"

namespace ov {
namespace nvidia_gpu {

class TileOp : public OperationBase {
public:
    TileOp(const CreationContext& context,
           const ov::Node& node,
           IndexCollection&& inputIds,
           IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;

private:
    kernel::TileParams params_{};
    size_t element_size_{};
};

}  // namespace nvidia_gpu
}  // namespace ov
