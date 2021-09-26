// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime.h>
#if defined __CUDACC__ && __has_include(<cuda_bf16.h>)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif
#include <cstdint>

namespace CUDAPlugin {
namespace kernel {

#define CUDAPLUGIN_LIST_TYPE_T_VALUES \
    X(undefined)                      \
    X(dynamic)                        \
    X(boolean)                        \
    X(bf16)                           \
    X(f16)                            \
    X(f32)                            \
    X(f64)                            \
    X(i4)                             \
    X(i8)                             \
    X(i16)                            \
    X(i32)                            \
    X(i64)                            \
    X(u1)                             \
    X(u4)                             \
    X(u8)                             \
    X(u16)                            \
    X(u32)                            \
    X(u64)

enum class Type_t {
#define X(val) val,
    CUDAPLUGIN_LIST_TYPE_T_VALUES
#undef X
};

template <Type_t>
struct cuda_type_traits {};

template <Type_t Type>
using cuda_type_traits_t = typename cuda_type_traits<Type>::value_type;

template <>
struct cuda_type_traits<Type_t::boolean> {
    using value_type = char;
};

template <>
struct cuda_type_traits<Type_t::bf16> {
#if defined __CUDACC__ && __has_include(<cuda_bf16.h>)
    using value_type = __nv_bfloat16;  // 8bit exponent, 7bit mantissa
#else
    using value_type = uint16_t;
#endif
};

template <>
struct cuda_type_traits<Type_t::f16> {
#if defined __CUDACC__ && __has_include(<cuda_bf16.h>)
    using value_type = __half;  // 1 sign bit, 5 exponent bits, and 10 mantissa bits.
#else
    using value_type = uint16_t;
#endif
};

template <>
struct cuda_type_traits<Type_t::f32> {
    using value_type = float;
};

template <>
struct cuda_type_traits<Type_t::f64> {
    using value_type = double;
};

template <>
struct cuda_type_traits<Type_t::i4> {
    using value_type = int8_t;
};

template <>
struct cuda_type_traits<Type_t::i8> {
    using value_type = int8_t;
};

template <>
struct cuda_type_traits<Type_t::i16> {
    using value_type = int16_t;
};

template <>
struct cuda_type_traits<Type_t::i32> {
    using value_type = int32_t;
};

template <>
struct cuda_type_traits<Type_t::i64> {
    using value_type = int64_t;
};

template <>
struct cuda_type_traits<Type_t::u1> {
    using value_type = int8_t;
};

template <>
struct cuda_type_traits<Type_t::u4> {
    using value_type = int8_t;
};

template <>
struct cuda_type_traits<Type_t::u8> {
    using value_type = uint8_t;
};

template <>
struct cuda_type_traits<Type_t::u16> {
    using value_type = uint16_t;
};

template <>
struct cuda_type_traits<Type_t::u32> {
    using value_type = uint32_t;
};

template <>
struct cuda_type_traits<Type_t::u64> {
    using value_type = uint64_t;
};

}  // namespace kernel
}  // namespace CUDAPlugin
