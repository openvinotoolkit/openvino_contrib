// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_operation_base.hpp"
#include "kernels/scatter_nd_update.hpp"

namespace ov {
namespace nvidia_gpu {

class ScatterNDUpdateOp : public OperationBase {
public:
    ScatterNDUpdateOp(const CreationContext& context,
                      const ov::Node& node,
                      IndexCollection&& inputIds,
                      IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override;
    WorkbufferRequest GetWorkBufferRequest() const override;
    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override;

private:
    std::optional<kernel::ScatterNDUpdate> kernel_;
    std::vector<size_t> input_data_dim_pading_;
};

}  // namespace nvidia_gpu
}  // namespace ov
