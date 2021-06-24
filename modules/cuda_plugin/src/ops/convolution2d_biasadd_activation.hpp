// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <memory>
#include <transformer/nodes/convolution2d_biasadd_activation.hpp>

namespace CUDAPlugin {

class Convolution2DBiasAddActivationOp : public OperationCuDnn {
 public:
  using NodeOp = CUDAPlugin::nodes::Conv2DBiasAddActivation;
  Convolution2DBiasAddActivationOp(const NodeOp& node, IndexCollection&& inputIds,
                                 IndexCollection&& outputIds);
  void Execute(const InferenceRequestContext& context, Inputs inputTensors,
               Outputs outputTensors, const Workbuffers& workbuffers) override;

  struct ArgIndices {
    static constexpr size_t input = 0;
    static constexpr size_t filter = 1;
    static constexpr size_t bias = 2;
    static constexpr size_t output = 0;
  };

 private:
  std::unique_ptr<IOperationExec> impl_;
};

}  // namespace CUDAPlugin
