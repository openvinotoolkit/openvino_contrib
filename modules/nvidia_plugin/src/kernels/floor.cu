// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/math.cuh>

#include "floor.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
struct FloorOpImpl {
    __device__ static inline T op(T x) { return CUDA::math::floor(x); }
};

Floor::Floor(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
    : ewu_{element_type, max_threads_per_block, num_elements} {}

void Floor::operator()(cudaStream_t stream, const void* in, void* out) const { ewu_(stream, in, out); }

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
