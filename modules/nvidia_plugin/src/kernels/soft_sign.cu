// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "soft_sign.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

namespace cumath = CUDA::math;

template <typename T>
__device__ constexpr T one = static_cast<T>(1);

template <typename T>
struct SoftSignOpImpl {
    __device__ static inline T op(T x) {
        return x / (one<T> + cumath::abs(x));
    }
};

template <>
struct SoftSignOpImpl<__nv_bfloat16> {
    __device__ static inline __nv_bfloat16 op(__nv_bfloat16 x) {
        return x / (__nv_bfloat16(1.0f) + cumath::abs(x));
    }
};

template <>
struct SoftSignOpImpl<__half> {
    __device__ static inline __half op(__half x) {
        return x / (__half(1.0f) + cumath::abs(x));
    }
};

SoftSign::SoftSign(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
    : impl_{element_type, max_threads_per_block, num_elements} {}

void SoftSign::operator()(cudaStream_t stream, const void* in0, void* out) const {
    impl_(stream, in0, out);
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
