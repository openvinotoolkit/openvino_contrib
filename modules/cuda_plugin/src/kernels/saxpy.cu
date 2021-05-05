// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

namespace CUDAPlugin {

__global__ void saxpy(int n, const float *a, const float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = *a * x[i] + y[i];
  }
}

} // namespace CUDAPlugin
