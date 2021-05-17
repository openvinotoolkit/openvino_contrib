// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime_api.h>
#include <cuda/stream.hpp>

namespace CUDAPlugin {

extern void sigmoid_run(CUDAPlugin::CudaStream& stream,
                        unsigned gridDim, unsigned blockDim,
                        size_t inputSize, const float *x, float *y);

} // namespace CUDAPlugin
