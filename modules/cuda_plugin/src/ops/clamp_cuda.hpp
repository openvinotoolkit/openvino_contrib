// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <ngraph/op/clamp.hpp>

#include "kernels/clamp.hpp"

namespace CUDAPlugin {

class ClampCudaOp : public OperationBase {
public:
    using NodeOp = ngraph::op::Clamp;

    ClampCudaOp(const CreationContext& context,
                const NodeOp& node,
                IndexCollection&& inputIds,
                IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

private:
    const size_t num_elements_;
    const double min_;
    const double max_;
    std::optional<kernel::Clamp> kernel_;
};

}  // namespace CUDAPlugin
