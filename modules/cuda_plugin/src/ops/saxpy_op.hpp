// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

namespace CUDAPlugin {

class SaxpyOp : public OperationBase {
 public:
    using OperationBase::OperationBase;
    void Execute(const InferenceRequestContext& context,
               Inputs inputTensors,
               Outputs outputTensors,
               const Workbuffers& workbuffers) override;
};

} // namespace CUDAPlugin
