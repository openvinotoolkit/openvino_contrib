// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn_ops_infer.h>

#include <cuda/device_pointers.hpp>
#include <cuda_operation_base.hpp>
#include <kernels/concat.hpp>
#include <openvino/op/concat.hpp>

namespace ov {
namespace nvidia_gpu {

class ConcatOp : public OperationBase {
public:
    using NodeOp = ov::op::v0::Concat;
    ConcatOp(const CreationContext& context,
             const NodeOp& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;
    WorkbufferRequest GetWorkBufferRequest() const override;
    void InitSharedImmutableWorkbuffers(const Buffers&) override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;

private:
    size_t immutableWbSize() const { return concat_kernel_.value().immutableWbSize(); }
    size_t mutableWbSize() const { return concat_kernel_.value().mutableWbSize(); }

    std::size_t num_inputs_;
    std::optional<kernel::Concat> concat_kernel_;
};

}  // namespace nvidia_gpu
}  // namespace ov
