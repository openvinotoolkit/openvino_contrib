// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "pooling_impl.hpp"

namespace CUDAPlugin {

class AvgPoolOp : public OperationBase {
 public:
  using OperationBase::OperationBase;
  explicit AvgPoolOp(const std::shared_ptr<ngraph::Node>& node,
                     std::vector<unsigned>&& inputIds,
                     std::vector<unsigned>&& outputIds);
  void Execute(const InferenceRequestContext& context, Inputs inputTensors,
               Outputs outputTensors) override;

 private:
  PoolingImpl impl_;
};

}  // namespace CUDAPlugin
