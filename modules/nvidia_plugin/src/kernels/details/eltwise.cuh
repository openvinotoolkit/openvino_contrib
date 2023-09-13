// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cuda_runtime.h>

#if !defined(__CUDACC__)
#include "openvino/core/shape.hpp"
#endif

namespace ov {
namespace nvidia_gpu {

namespace eltwise {

/*
 * Holds information about blocks and threads hierarchy
 * */
struct KernelExecAttrs {
    dim3 grid;
    dim3 block;
    unsigned elementsPerThread;
#if !defined(__CUDACC__)
    KernelExecAttrs(const ov::Shape&, unsigned threadsPerBlock, unsigned elementsPerThread);
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

template <int WorkloadRank>
__device__ unsigned index(const std::size_t shape[WorkloadRank], unsigned elementsPerThread) {
    return ((blockIdx.x * gridDim.y + blockIdx.y) * shape[WorkloadRank - 1] + blockIdx.z * blockDim.x + threadIdx.x) *
           elementsPerThread;
}

template <>
__device__ unsigned index<1>(const std::size_t shape[1], unsigned elementsPerThread) {
    return (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
}

template <>
__device__ unsigned index<2>(const std::size_t shape[2], unsigned elementsPerThread) {
    return (blockIdx.x * shape[1] + blockIdx.y * blockDim.x + threadIdx.x) * elementsPerThread;
}

/*
 * Calculates an index of workload of 'shape' shape in the given dimension in the current kernel invocation.
 * The kernel should be launched with parameters calculated in KernelExecAttrs.
 * */
template <int WorkloadRank>
static inline __device__ int index_in_dim(int dim, const std::size_t shape[WorkloadRank], unsigned elementsPerThread);

template <>
__device__ int index_in_dim<1>(int dim, const std::size_t shape[1], unsigned elementsPerThread) {
    switch (dim) {
        case 0:
            return (blockIdx.x * blockDim.x + threadIdx.x) * elementsPerThread;
    }
    return {};
}

template <>
__device__ int index_in_dim<2>(int dim, const std::size_t shape[2], unsigned elementsPerThread) {
    switch (dim) {
        case 0:
            return blockIdx.x;
        case 1:
            return (blockIdx.y * blockDim.x + threadIdx.x) * elementsPerThread;
    }
    return {};
}

template <>
__device__ int index_in_dim<3>(int dim, const std::size_t shape[3], unsigned elementsPerThread) {
    switch (dim) {
        case 0:
            return blockIdx.x;
        case 1:
            return blockIdx.y;
        case 2:
            return (blockIdx.z * blockDim.x + threadIdx.x) * elementsPerThread;
    }
    return {};
}

template <>
__device__ int index_in_dim<4>(int dim, const std::size_t shape[4], unsigned elementsPerThread) {
    switch (dim) {
        case 0:
            return blockIdx.x / shape[1];
        case 1:
            return blockIdx.x % shape[1];
        case 2:
            return blockIdx.y;
        case 3:
            return (blockIdx.z * blockDim.x + threadIdx.x) * elementsPerThread;
    }
    return {};
}

template <>
__device__ int index_in_dim<5>(int dim, const std::size_t shape[5], unsigned elementsPerThread) {
    switch (dim) {
        case 0:
            return blockIdx.x / shape[1];
        case 1:
            return blockIdx.x % shape[1];
        case 2:
            return blockIdx.y / shape[3];
        case 3:
            return blockIdx.y % shape[3];
        case 4:
            return (blockIdx.z * blockDim.x + threadIdx.x) * elementsPerThread;
    }
    return {};
}

#endif

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
