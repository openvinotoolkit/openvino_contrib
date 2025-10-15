// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "components/numpy_broadcast_params.h"
#include "cuda_operation_base.hpp"
#include "kernels/fake_quantize.hpp"
#include "openvino/op/fake_quantize.hpp"

namespace ov {
namespace nvidia_gpu {

class FakeQuantizeOp : public OperationBase {
public:
    using NodeOp = ov::op::v0::FakeQuantize;
    FakeQuantizeOp(const CreationContext& context,
                   const NodeOp& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds);

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;

private:
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override final;

    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override final;
    WorkbufferRequest GetWorkBufferRequest() const override final;

private:
    std::unique_ptr<NumpyBroadcastParams> in_low_broadcast_params_;
    std::unique_ptr<NumpyBroadcastParams> in_high_broadcast_params_;
    std::unique_ptr<NumpyBroadcastParams> out_low_broadcast_params_;
    std::unique_ptr<NumpyBroadcastParams> out_high_broadcast_params_;

    std::vector<WorkbufferRequest::size_in_bytes_t> immutable_buffer_sizes_;
    std::optional<kernel::FakeQuantize> kernel_;
};

}  // namespace nvidia_gpu
}  // namespace ov
