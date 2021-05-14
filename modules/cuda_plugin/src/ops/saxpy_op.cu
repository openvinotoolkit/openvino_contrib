// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <kernels/saxpy.cuh>
#include <cuda_operation_registry.hpp>

#include "saxpy_op.hpp"

namespace CUDAPlugin {

void SaxpyOp::Execute(const InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors) {
    size_t size = 10000;
    const unsigned maxBlockSize = context.getThreadContext().device().props().maxThreadsPerBlock;
    const unsigned gridSize = size / maxBlockSize;
    const unsigned blockSize = gridSize > 1 ? maxBlockSize : size % maxBlockSize;
    const dim3 gridDim = dim3{gridSize ? gridSize : 1};
    const dim3 blockDim = dim3{blockSize ? blockSize : maxBlockSize};
    saxpy<<<gridDim, blockDim>>>(
        size,
        inputTensors[0].cast<const float*>().get(),
        inputTensors[1].cast<const float*>().get(),
        outputTensors[0].cast<float*>().get());
}

OPERATION_REGISTER(SaxpyOp, "Saxpy");

} // namespace CUDAPlugin
