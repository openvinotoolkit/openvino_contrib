// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cuda_runtime.h>

#if not defined(__CUDACC__)
#include <ngraph/shape.hpp>
#endif

namespace CUDAPlugin {

namespace eltwise {

/*
 * Holds information about blocks and threads hierarchy
 * */
struct KernelExecAttrs {
    dim3 grid;
    dim3 block;
    unsigned elementsPerThread;
#if not defined(__CUDACC__)
    KernelExecAttrs(const ngraph::Shape&, unsigned threadsPerBlock, unsigned elementsPerThread);
#endif
};

}  // namespace eltwise

namespace kernel {

#if defined(__CUDACC__)

/*
 * Calculates index in workload of 'shape' shape in the current kernel invocation.
 * The kernel should be launched with parameters calculated in KernelExecAttrs.
 * */
template <int WorkloadRank>
static inline __device__ unsigned index(const std::size_t shape[WorkloadRank], unsigned elementsPerThread);

template <>
__device__ unsigned index<1>(const std::size_t shape[1], unsigned elementsPerThread) {
    return (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
}

#endif

}  // namespace kernel
}  // namespace CUDAPlugin
