// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_operation_base.hpp"
#include "kernels/interpolate_linear.hpp"
#include "openvino/op/interpolate.hpp"

namespace ov {
namespace nvidia_gpu {

class InterpolateLinearOp : public OperationBase {
public:
    using NodeOp = ov::op::v4::Interpolate;

    InterpolateLinearOp(const CreationContext& context,
                        const NodeOp& stridedSliceOp,
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
    std::optional<kernel::InterpolateLinear> interpolate_;
};

}  // namespace nvidia_gpu
}  // namespace ov
