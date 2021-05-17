// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sigmoid.hpp"

namespace CUDAPlugin {

static __global__ void sigmoid(const size_t inputSize, const float *x, float *y) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inputSize) {
        y[i] = 1 / (1 + expf(-x[i]));
    }
}

void sigmoid_run(CUDAPlugin::CudaStream& stream,
                 const unsigned gridDim, const unsigned blockDim,
                 size_t inputSize, const float *x, float *y) {
    stream.runKernel(gridDim, blockDim, sigmoid, inputSize, x, y);
}

} // namespace CUDAPlugin