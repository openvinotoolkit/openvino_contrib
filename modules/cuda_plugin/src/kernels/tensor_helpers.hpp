// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime_api.h>

namespace CUDAPlugin {
namespace kernel {

template <typename T, unsigned N>
using Shape = T[N];

template <typename T, unsigned N>
__host__ __device__ size_t rank(const Shape<T, N>& shape) {
    size_t rank = 0;
    for (auto& dim : shape) {
        if (dim == 0) {
            break;
        }
        rank++;
    }
    return rank;
}

template <typename T, unsigned N>
inline __host__ __device__ void shape_indices(const Shape<T, N>& shape,
                                              const size_t flatAddress,
                                              Shape<T, N>& indexes) {
    const size_t shape_rank = rank(shape);
    size_t last_iter = flatAddress;
    for (int dim = shape_rank - 1; dim >= 0; --dim) {
        if (dim != 0) {
            indexes[dim] = last_iter % shape[dim];
            last_iter = last_iter / shape[dim];
        } else {
            indexes[dim] = last_iter;
        }
    }
}

template <typename T, unsigned N>
inline __host__ __device__ size_t flat_address(const Shape<T, N>& shape, const Shape<T, N>& indexes) {
    size_t address = 0;
    const size_t shape_rank = rank(shape);
    size_t mul_shapes = 1;
    const size_t startDim = shape_rank - 1;
    for (int dim = startDim; dim >= 0; --dim) {
        if (dim != startDim) {
            address += indexes[dim] * mul_shapes;
        } else {
            address += indexes[dim];
        }
        mul_shapes *= shape[dim];
    }
    return address;
}

template <typename T, unsigned N>
inline size_t shape_size(const Shape<T, N>& shape) {
    size_t size = 1;
    for (auto d : shape) {
        if (d == 0) {
            break;
        }
        size *= d;
    }
    return size;
}

inline std::pair<unsigned, unsigned> calculateElementwiseGrid(const size_t size, const size_t max_threads_per_block) {
    const auto num_blocks =
        (size % max_threads_per_block == 0) ? (size / max_threads_per_block) : (size / max_threads_per_block + 1);
    const auto threads_per_block = (num_blocks == 1) ? size : max_threads_per_block;
    return {num_blocks, threads_per_block};
}

}  // namespace kernel
}  // namespace CUDAPlugin
