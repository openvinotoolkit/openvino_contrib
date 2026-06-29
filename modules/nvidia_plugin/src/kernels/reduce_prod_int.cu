// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_prod_int.cuh"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

/// Single-block reduction kernel for small integer arrays.
/// Uses shared memory for the reduction. Handles up to 1024 elements.
static __global__ void reduce_prod_int32_kernel(const int32_t* input,
                                                int32_t* output,
                                                size_t num_elements) {
    extern __shared__ int32_t sdata[];

    unsigned int tid = threadIdx.x;
    // Each thread loads one element (or identity = 1 for out-of-range)
    sdata[tid] = (tid < num_elements) ? input[tid] : 1;
    __syncthreads();

    // Parallel reduction using multiplication
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sdata[0];
    }
}

void reduce_prod_int32(cudaStream_t stream,
                       const int32_t* input,
                       int32_t* output,
                       size_t num_elements) {
    if (num_elements == 0) {
        return;
    }
    // Round up to next power of 2 for shared memory reduction
    unsigned int threads = 1;
    while (threads < num_elements) {
        threads <<= 1;
    }
    if (threads > 1024) threads = 1024;  // CUDA max threads per block

    reduce_prod_int32_kernel<<<1, threads, threads * sizeof(int32_t), stream>>>(
        input, output, num_elements);
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
