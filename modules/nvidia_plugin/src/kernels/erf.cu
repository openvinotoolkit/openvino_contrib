// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/math.cuh>

#include "erf.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
struct ErfOpImpl {
    __device__ static inline T op(T x) { return CUDA::math::erff(x); }
};

Erf::Erf(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
    : ewu_{element_type, max_threads_per_block, num_elements} {}

void Erf::operator()(cudaStream_t stream, const void* in, void* out) const { ewu_(stream, in, out); }

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
