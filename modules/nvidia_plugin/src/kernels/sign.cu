// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sign.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

namespace cumath = CUDA::math;

template <typename T, typename Enable = void>
struct SignOpImpl {
    __device__ static inline T op(T x);
};

template <>
struct SignOpImpl<char> {
    __device__ static inline char op(char x) {
        return cumath::sign_int(x);
    }
};

template <typename T>
struct SignOpImpl<T, typename std::enable_if<std::is_integral<T>::value &&
                                             !std::is_unsigned<T>::value>::type> {
    __device__ static inline T op(T x) {
        return cumath::sign_int(x);
    }
};

template <typename T>
struct SignOpImpl<T, typename std::enable_if<std::is_integral<T>::value &&
                                             std::is_unsigned<T>::value>::type> {
    __device__ static inline T op(T x) {
        return cumath::sign_uint(x);
    }
};

template <typename T>
struct SignOpImpl<T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
    __device__ static inline T op(T x) {
        return cumath::sign_float(x);
    }
};

template <>
struct SignOpImpl<__nv_bfloat16> {
    __device__ static inline __nv_bfloat16 op(__nv_bfloat16 x) {
        return cumath::sign_float(x);
    }
};

template <>
struct SignOpImpl<__half> {
    __device__ static inline __half op(__half x) {
        return cumath::sign_float(x);
    }
};

Sign::Sign(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
    : impl_{element_type, max_threads_per_block, num_elements} {}

void Sign::operator()(cudaStream_t stream, const void* in0, void* out) const {
    impl_(stream, in0, out);
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
