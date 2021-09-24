// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

namespace CUDAPlugin {

class SaxpyOp : public OperationBase {
public:
    SaxpyOp(const CUDA::CreationContext& context,
              const std::shared_ptr<ngraph::Node>& node,
              IndexCollection&& inputIds,
              IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

private:
    static constexpr size_t kSize = 10000;
    dim3 grid_dim_;
    dim3 block_dim_;
};

} // namespace CUDAPlugin
