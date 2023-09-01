// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/float16.hpp>

#include "details/tensor_helpers.hpp"
#include "swish.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

namespace cumath = CUDA::math;
template <typename T>
struct SwishOpImpl {
    __device__ static inline T op(T x, double beta) {
        return x / (static_cast<T>(1.0f) + cumath::exp(-x * static_cast<T>(beta)));
    }
};

Swish::Swish(Type_t element_type, size_t max_threads_per_block, size_t num_elements, double beta)
    : ewu_{element_type, max_threads_per_block, num_elements}, beta_{beta} {}

void Swish::operator()(cudaStream_t stream, const void* in, void* out) const { ewu_(stream, in, out, beta_); }

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
