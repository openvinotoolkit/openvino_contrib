// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

#if CUDA_VERSION >= 11000
#define CUDA_HAS_BF16_TYPE
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530
#define CUDA_HAS_HALF_MATH
#endif  // (__CUDA_ARCH__ >= 530) || !defined(__CUDA_ARCH__)

#if defined(CUDA_HAS_BF16_TYPE) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
#define CUDA_HAS_BF16_MATH
#endif  // defined (CUDA_HAS_BF16_TYPE) && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))

#if defined(__CUDACC__) && CUDA_VERSION < 12000
#if !defined(CUDA_HAS_HALF_MATH)
/* Some basic arithmetic operations expected of a builtin */
__device__ __forceinline__ __half operator+(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) + static_cast<float>(rh);
}
__device__ __forceinline__ __half operator-(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) - static_cast<float>(rh);
}
__device__ __forceinline__ __half operator*(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) * static_cast<float>(rh);
}
__device__ __forceinline__ __half operator/(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) / static_cast<float>(rh);
}

__device__ __forceinline__ __half &operator+=(__half &lh, const __half &rh) {
    lh = static_cast<float>(lh) + static_cast<float>(rh);
    return lh;
}
__device__ __forceinline__ __half &operator-=(__half &lh, const __half &rh) {
    lh = static_cast<float>(lh) - static_cast<float>(rh);
    return lh;
}
__device__ __forceinline__ __half &operator*=(__half &lh, const __half &rh) {
    lh = static_cast<float>(lh) * static_cast<float>(rh);
    return lh;
}
__device__ __forceinline__ __half &operator/=(__half &lh, const __half &rh) {
    lh = static_cast<float>(lh) / static_cast<float>(rh);
    return lh;
}

/* Note for increment and decrement we use the raw value 0x3C00U equating to half(1.0F), to avoid the extra conversion
 */
__device__ __forceinline__ __half &operator++(__half &h) {
    __half_raw one;
    one.x = 0x3C00U;
    h += one;
    return h;
}
__device__ __forceinline__ __half &operator--(__half &h) {
    __half_raw one;
    one.x = 0x3C00U;
    h -= one;
    return h;
}
__device__ __forceinline__ __half operator++(__half &h, const int ignored) {
    const __half ret = h;
    __half_raw one;
    one.x = 0x3C00U;
    h += one;
    return ret;
}
__device__ __forceinline__ __half operator--(__half &h, const int ignored) {
    const __half ret = h;
    __half_raw one;
    one.x = 0x3C00U;
    h -= one;
    return ret;
}

/* Unary plus and inverse operators */
__device__ __forceinline__ __half operator+(const __half &h) { return h; }
__device__ __forceinline__ __half operator-(const __half &h) { return static_cast<float>(h); }

/* Some basic comparison operations to make it look like a builtin */
__device__ __forceinline__ bool operator==(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) == static_cast<float>(rh);
}
__device__ __forceinline__ bool operator!=(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) != static_cast<float>(rh);
}
__device__ __forceinline__ bool operator>(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) > static_cast<float>(rh);
}
__device__ __forceinline__ bool operator<(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) < static_cast<float>(rh);
}
__device__ __forceinline__ bool operator>=(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) >= static_cast<float>(rh);
}
__device__ __forceinline__ bool operator<=(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) <= static_cast<float>(rh);
}
#endif /* !defined(CUDA_HAS_HALF_MATH) || defined(__CUDA_NO_HALF_OPERATORS__) */

#if defined(CUDA_HAS_BF16_TYPE) && !defined(CUDA_HAS_BF16_MATH)
/* Some basic arithmetic operations expected of a builtin */
__device__ __forceinline__ __nv_bfloat16 operator+(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
    return static_cast<float>(lh) + static_cast<float>(rh);
}
__device__ __forceinline__ __nv_bfloat16 operator-(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
    return static_cast<float>(lh) - static_cast<float>(rh);
}
__device__ __forceinline__ __nv_bfloat16 operator*(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
    return static_cast<float>(lh) * static_cast<float>(rh);
}
__device__ __forceinline__ __nv_bfloat16 operator/(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
    return static_cast<float>(lh) / static_cast<float>(rh);
}

__device__ __forceinline__ __nv_bfloat16 &operator+=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
    lh = static_cast<float>(lh) + static_cast<float>(rh);
    return lh;
}
__device__ __forceinline__ __nv_bfloat16 &operator-=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
    lh = static_cast<float>(lh) - static_cast<float>(rh);
    return lh;
}
__device__ __forceinline__ __nv_bfloat16 &operator*=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
    lh = static_cast<float>(lh) * static_cast<float>(rh);
    return lh;
}
__device__ __forceinline__ __nv_bfloat16 &operator/=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
    lh = static_cast<float>(lh) / static_cast<float>(rh);
    return lh;
}

/* Note for increment and decrement we use the raw value 0x3C00U equating to half(1.0F), to avoid the extra conversion
 */
__device__ __forceinline__ __nv_bfloat16 &operator++(__nv_bfloat16 &h) {
    __nv_bfloat16_raw one;
    one.x = 0x3C00U;
    h += one;
    return h;
}
__device__ __forceinline__ __nv_bfloat16 &operator--(__nv_bfloat16 &h) {
    __nv_bfloat16_raw one;
    one.x = 0x3C00U;
    h -= one;
    return h;
}
__device__ __forceinline__ __nv_bfloat16 operator++(__nv_bfloat16 &h, const int ignored) {
    const __nv_bfloat16 ret = h;
    __nv_bfloat16_raw one;
    one.x = 0x3C00U;
    h += one;
    return ret;
}
__device__ __forceinline__ __nv_bfloat16 operator--(__nv_bfloat16 &h, const int ignored) {
    const __nv_bfloat16 ret = h;
    __nv_bfloat16_raw one;
    one.x = 0x3C00U;
    h -= one;
    return ret;
}

/* Unary plus and inverse operators */
__device__ __forceinline__ __nv_bfloat16 operator+(const __nv_bfloat16 &h) { return h; }
__device__ __forceinline__ __nv_bfloat16 operator-(const __nv_bfloat16 &h) { return static_cast<float>(h); }

/* Some basic comparison operations to make it look like a builtin */
__device__ __forceinline__ bool operator==(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
    return static_cast<float>(lh) == static_cast<float>(rh);
}
__device__ __forceinline__ bool operator!=(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
    return static_cast<float>(lh) != static_cast<float>(rh);
}
__device__ __forceinline__ bool operator>(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
    return static_cast<float>(lh) > static_cast<float>(rh);
}
__device__ __forceinline__ bool operator<(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
    return static_cast<float>(lh) < static_cast<float>(rh);
}
__device__ __forceinline__ bool operator>=(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
    return static_cast<float>(lh) >= static_cast<float>(rh);
}
__device__ __forceinline__ bool operator<=(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) {
    return static_cast<float>(lh) <= static_cast<float>(rh);
}
#endif  /* defined(CUDA_HAS_BF16_TYPE) && !defined(CUDA_HAS_BF16_MATH) */
#endif  // defined(__CUDACC__) && CUDA_VERSION < 12000
