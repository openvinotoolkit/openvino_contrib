// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/runtime.hpp>

namespace CUDAPlugin {

void sigmoid_run(const CUDA::Stream &stream, unsigned gridDim,
                 unsigned blockDim, size_t inputSize, const float *x, float *y);

} // namespace CUDAPlugin
