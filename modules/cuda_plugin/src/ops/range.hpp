// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <optional>

#include "kernels/range.hpp"

namespace CUDAPlugin {

class RangeOp : public OperationBase {
    enum InputIdx { START_INDX, STOP_INDX, STEP_INDX };

public:
    RangeOp(const CreationContext& context,
            const ngraph::Node& node,
            IndexCollection&& inputIds,
            IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers& workbuffers) const override;

private:
    size_t output_size_;
    std::optional<kernel::RangeKernelOp> kernel_op_;
};

}  // namespace CUDAPlugin
