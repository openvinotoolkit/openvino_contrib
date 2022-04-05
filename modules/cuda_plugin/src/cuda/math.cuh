// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "float16.hpp"

namespace CUDA {
namespace math {

template <typename T>
inline __device__ T min(T x, T y) {
    return x < y ? x : y;
}

template <typename T>
inline __device__ T max(T x, T y) {
    return x > y ? x : y;
}

template <typename T>
inline __device__ T exp(T x) {
    return ::exp(x);
}

#if defined(CUDA_HAS_HALF_MATH)
template <>
inline __device__ __half exp<__half>(__half x) {
    return ::hexp(x);
}
#else
template <>
inline __device__ __half min<__half>(__half x, __half y) {
    return min<float>(static_cast<float>(x), static_cast<float>(y));
}

template <>
inline __device__ __half max<__half>(__half x, __half y) {
    return max<float>(static_cast<float>(x), static_cast<float>(y));
}

template <>
inline __device__ __half exp<__half>(__half x) {
    return exp(static_cast<float>(x));
}
#endif  // !defined (CUDA_HAS_HALF_MATH)

#if defined(CUDA_HAS_BF16_TYPE)
#if defined(CUDA_HAS_BF16_MATH)
template <>
inline __device__ __nv_bfloat16 exp<__nv_bfloat16>(__nv_bfloat16 x) {
    return ::hexp(x);
}
#else
template <>
inline __device__ __nv_bfloat16 min<__nv_bfloat16>(__nv_bfloat16 x, __nv_bfloat16 y) {
    return min<float>(static_cast<float>(x), static_cast<float>(y));
}

template <>
inline __device__ __nv_bfloat16 max<__nv_bfloat16>(__nv_bfloat16 x, __nv_bfloat16 y) {
    return max<float>(static_cast<float>(x), static_cast<float>(y));
}

template <>
inline __device__ __nv_bfloat16 exp<__nv_bfloat16>(__nv_bfloat16 x) {
    return exp<float>(static_cast<float>(x));
}
#endif  // defined (CUDA_HAS_BF16_MATH)
#endif  // defined (CUDA_HAS_BF16_TYPE)

}  // namespace math
}  // namespace CUDA
