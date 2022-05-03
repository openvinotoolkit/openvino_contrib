// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/fake_quantize.hpp>

#include "components/numpy_broadcast_params.h"
#include "cuda_operation_base.hpp"
#include "kernels/fake_quantize.hpp"

namespace CUDAPlugin {

class FakeQuatizeOp : public OperationBase {
public:
    using NodeOp = ngraph::op::FakeQuantize;
    FakeQuatizeOp(const CreationContext& context,
                  const NodeOp& node,
                  IndexCollection&& inputIds,
                  IndexCollection&& outputIds);

private:
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const final;

    void InitSharedImmutableWorkbuffers(const Buffers& buffers) final;
    WorkbufferRequest GetWorkBufferRequest() const final;

private:
    std::unique_ptr<NumpyBroadcastParams> in_low_broadcast_params_;
    std::unique_ptr<NumpyBroadcastParams> in_high_broadcast_params_;
    std::unique_ptr<NumpyBroadcastParams> out_low_broadcast_params_;
    std::unique_ptr<NumpyBroadcastParams> out_high_broadcast_params_;

    std::vector<WorkbufferRequest::size_in_bytes_t> immutable_buffer_sizes_;
    std::optional<kernel::FakeQuantize> kernel_;
};

}  // namespace CUDAPlugin
