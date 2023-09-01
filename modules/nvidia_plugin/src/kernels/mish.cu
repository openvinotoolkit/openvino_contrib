// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mish.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

namespace cumath = CUDA::math;

template <typename T>
struct MishOpImpl {
    __device__ static inline T softplus(T x) {
        return cumath::log(static_cast<T>(1.0) + cumath::exp(x));
    }

    __device__ static inline T op(T x) {
        return x * cumath::tanh(softplus(x));
    }
};

Mish::Mish(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
    : impl_{element_type, max_threads_per_block, num_elements} {}

void Mish::operator()(cudaStream_t stream, const void* in0, void* out) const {
    impl_(stream, in0, out);
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
