// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_operation_base.hpp"
#include "kernels/broadcast.hpp"
#include "ngraph/op/broadcast.hpp"

namespace CUDAPlugin {

class BroadcastOp : public OperationBase {
public:
    using NodeOp = ngraph::op::v3::Broadcast;
    BroadcastOp(const CreationContext& context,
                const NodeOp& stridedSliceOp,
                IndexCollection&& inputIds,
                IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers& workbuffers) const override;

    WorkbufferRequest GetWorkBufferRequest() const override;
    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override;

private:
    std::vector<size_t> broadcast_dims_;
    std::vector<size_t> src_strides_;
    std::vector<size_t> dst_strides_;

    std::optional<kernel::Broadcast> kernel_;
};

}  // namespace CUDAPlugin
