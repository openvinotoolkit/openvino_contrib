// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_operation_base.hpp"
#include "kernels/broadcast.hpp"
#include "ngraph/op/broadcast.hpp"
#include "components/numpy_broadcast_params.h"

namespace CUDAPlugin {

class BroadcastOp : public OperationBase {
public:
    using NodeOp = ov::op::v3::Broadcast;
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
    std::vector<WorkbufferRequest::size_in_bytes_t> immutable_buffer_sizes_;
    std::unique_ptr<NumpyBroadcastParams> broadcast_params_;

    std::optional<kernel::Broadcast> kernel_;
};

}  // namespace CUDAPlugin
