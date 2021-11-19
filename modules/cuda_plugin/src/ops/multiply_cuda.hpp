// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "kernels/elementwise.hpp"

namespace CUDAPlugin {

class MultiplyCudaOp : public OperationBase {
public:
    MultiplyCudaOp(const CreationContext& context,
                   const ngraph::Node& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

private:
    size_t in0_num_elements_;
    size_t in1_num_elements_;
    std::optional<kernel::Elementwise> kernel_;
};

}  // namespace CUDAPlugin
