// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/device_pointers.hpp>
#include <cuda_operation_base.hpp>

namespace CUDAPlugin {

class SigmoidOp : public OperationBase {
public:
    SigmoidOp(const CreationContext& context,
              const std::shared_ptr<ngraph::Node>& node,
              IndexCollection&& inputIds,
              IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) override;

private:
    size_t input_size_;
    size_t output_size_;
    unsigned num_blocks_;
    unsigned threads_per_block_;
};

}  // namespace CUDAPlugin
