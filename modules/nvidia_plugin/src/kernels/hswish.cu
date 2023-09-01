// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "hswish.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

namespace cumath = CUDA::math;

template <typename T>
struct HSwishOpImpl {
    __device__ static inline T op(T x) {
        return x * cumath::min(cumath::max(x + static_cast<T>(3.0), static_cast<T>(0.0)),
                               static_cast<T>(6.0)) * static_cast<T>(1.0 / 6.0);
    }
};

HSwish::HSwish(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
    : impl_{element_type, max_threads_per_block, num_elements} {}

void HSwish::operator()(cudaStream_t stream, const void* in0, void* out) const {
    impl_(stream, in0, out);
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
