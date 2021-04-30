// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <kernels/saxpy.cuh>

#include "saxpy_op.hpp"

namespace CUDAPlugin {

void SaxpyOp::Execute(const InferenceRequestContext& context,
                      gsl::span<const void*> inputTensors,
                      gsl::span<void*> outputTensors) {
  size_t size = 10000;
  auto gridNum = size / 1024;
  auto blockNum = gridNum > 1 ? 1024 : size / 1024;
  saxpy<<<gridNum, blockNum>>>(size,
                               *reinterpret_cast<const float*>(inputTensors[0]),
                               reinterpret_cast<const float*>(inputTensors[1]),
                               reinterpret_cast<float*>(outputTensors[0]));
}

} // namespace CUDAPlugin
