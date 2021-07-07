// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "pooling_impl.hpp"

namespace CUDAPlugin {

class MaxPoolOp : public OperationCuDnn {
 public:
  explicit MaxPoolOp(const CUDA::Device& device,
                     const std::shared_ptr<ngraph::Node>& node,
                     std::vector<unsigned>&& inputIds,
                     std::vector<unsigned>&& outputIds);
  void Execute(const InferenceRequestContext& context, Inputs inputTensors,
               Outputs outputTensors, const Workbuffers& workbuffers) override;

 private:
  PoolingImpl impl_;
};

}  // namespace CUDAPlugin
