// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_operation_base.hpp"
#include "kernels/interpolate_cubic.hpp"
#include "ngraph/op/interpolate.hpp"

namespace CUDAPlugin {

class InterpolateCubicOp : public OperationBase {
public:
    using NodeOp = ngraph::op::v4::Interpolate;
    InterpolateCubicOp(const CreationContext& context,
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
    std::optional<kernel::InterpolateCubic> interpolate_;
};

}  // namespace CUDAPlugin
