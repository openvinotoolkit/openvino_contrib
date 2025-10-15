// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph.cuh"

__global__ void VecAdd(int* A, int* B, int* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void enqueueVecAdd(const CUDA::Stream s, dim3 gridDim, dim3 blockDim,
                   int* devA, int* devB, int* devC, int N) {
    s.run(gridDim, blockDim, VecAdd, devA, devB, devC, N);
}