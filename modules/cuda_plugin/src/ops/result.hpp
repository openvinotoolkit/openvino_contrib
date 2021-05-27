// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <gpu/device_pointers.hpp>

namespace CUDAPlugin {

class ResultOp : public OperationBase {
 public:
    ResultOp(const std::shared_ptr<ngraph::Node>& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors) override;

 private:
    std::string output_tensor_name_;
};

} // namespace CUDAPlugin
