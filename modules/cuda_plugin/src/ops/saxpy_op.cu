// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <kernels/saxpy.cuh>

#include "saxpy_op.hpp"

namespace CUDAPlugin {

void SaxpyOp::Execute(const InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors) {
  size_t size = 10000;
  auto gridNum = size / 1024;
  auto blockNum = gridNum > 1 ? 1024 : size / 1024;
  saxpy<<<gridNum, blockNum>>>(size,
                               inputTensors[0].cast<const float*>(),
                               inputTensors[1].cast<const float*>(),
                               outputTensors[0].cast<float*>());
}

} // namespace CUDAPlugin
