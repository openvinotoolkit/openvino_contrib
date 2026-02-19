// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/math.cuh>

#include "softplus.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
struct SoftPlusOpImpl {
    __device__ static inline T op(T x) {
        // Numerically stable: for large x, exp(x) overflows in float32 (x > ~88.7),
        // but softplus(x) ≈ x with negligible error. Threshold 20 matches PyTorch default
        // (torch.nn.Softplus) — at x=20 the error |softplus(x)-x| ≈ 2e-9 << float32 epsilon.
        return (x > T{20.0f}) ? x : CUDA::math::log(T{1.0f} + CUDA::math::exp(x));
    }
};

SoftPlus::SoftPlus(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
    : ewu_{element_type, max_threads_per_block, num_elements} {}

void SoftPlus::operator()(cudaStream_t stream, const void* in, void* out) const { ewu_(stream, in, out); }

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
