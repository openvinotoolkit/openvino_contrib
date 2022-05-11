// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "float16.hpp"

namespace CUDA {
namespace math {

/* =================== limit_min =================== */
template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, bool> = true>
inline __device__ __host__ T limit_min() {
    return 0;
}

template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, bool> = true>
inline __device__ __host__ T limit_min() {
    return static_cast<T>(1) << (sizeof(T) * 8 - 1);
}
/* ================================================= */

/* =================== limit_max =================== */
template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, bool> = true>
inline __device__ __host__ T limit_max() {
    return static_cast<T>(-1);
}

template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, bool> = true>
inline __device__ __host__ T limit_max() {
    return ~limit_min<T>();
}
/* ================================================= */

template <typename T>
inline __device__ T round(T x) {
    return ::round(x);
}

/* ===================== floor ===================== */
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline __device__ T floor(T x) {
    return x;
}

template <typename T, std::enable_if_t<!std::is_integral<T>::value, bool> = true>
inline __device__ T floor(T x) {
    return ::floor(x);
}

inline __device__ float floor(float x) { return ::floorf(x); }
/* ================================================= */

/* ===================== trunc ===================== */
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline __device__ T trunc(T x) {
    return x;
}

template <typename T, std::enable_if_t<!std::is_integral<T>::value, bool> = true>
inline __device__ T trunc(T x) {
    return ::trunc(x);
}

inline __device__ float trunc(float x) { return ::truncf(x); }
/* ================================================= */

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

template <typename T>
inline __device__ T pow(T x, T y) {
    return ::pow(x, y);
}

template <typename T>
inline __device__ T sqrt(T a) {
    return ::sqrt(a);
}

#ifdef __CUDACC__
/* ==================== __half ===================== */
template <>
inline __device__ __half round(__half x) {
    return ::round(static_cast<float>(x));
}

template <>
inline __device__ __half pow<__half>(__half x, __half y) {
    return powf(static_cast<float>(x), static_cast<float>(y));
}

#if defined(CUDA_HAS_HALF_MATH)
inline __device__ __half floor(__half x) { return ::hfloor(x); }

inline __device__ __half trunc(__half x) { return ::htrunc(x); }

template <>
inline __device__ __half exp<__half>(__half x) {
    return ::hexp(x);
}

template <>
inline __device__ __half sqrt<__half>(__half a) {
    return ::hsqrt(a);
}

#else   // defined (CUDA_HAS_HALF_MATH)

inline __device__ __half floor(__half x) { return floor(static_cast<float>(x)); }

inline __device__ __half trunc(__half x) { return trunc(static_cast<float>(x)); }

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

template <>
inline __device__ __half sqrt<__half>(__half a) {
    return ::sqrt(static_cast<float>(a));
}
#endif  // defined (CUDA_HAS_HALF_MATH)

/* ================================================= */

/* ================ __nv_bfloat16 ================== */
#if defined(CUDA_HAS_BF16_TYPE)

template <>
inline __device__ __nv_bfloat16 round(__nv_bfloat16 x) {
    return ::round(static_cast<float>(x));

}

#if defined(CUDA_HAS_BF16_MATH)
inline __device__ __nv_bfloat16 floor(__nv_bfloat16 x) { return ::hfloor(x); }

inline __device__ __nv_bfloat16 trunc(__nv_bfloat16 x) { return ::htrunc(x); }

template <>
inline __device__ __nv_bfloat16 exp<__nv_bfloat16>(__nv_bfloat16 x) {
    return ::hexp(x);
}

#else  // defined (CUDA_HAS_BF16_MATH)

inline __device__ __nv_bfloat16 floor(__nv_bfloat16 x) { return floor(static_cast<float>(x)); }

inline __device__ __nv_bfloat16 trunc(__nv_bfloat16 x) { return trunc(static_cast<float>(x)); }

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
/* ================================================= */
#endif  // __CUDACC__

}  // namespace math
}  // namespace CUDA
