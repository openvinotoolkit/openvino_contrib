// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <ngraph/op/swish.hpp>

#include "kernels/swish.hpp"

namespace CUDAPlugin {

class SwishOp : public OperationBase {
public:
    SwishOp(const CreationContext& context,
            const ngraph::Node& node,
            IndexCollection&& inputIds,
            IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

private:
    double beta_;
    size_t num_elements_;
    std::optional<kernel::Swish> kernel_;
};

}  // namespace CUDAPlugin
