// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cuda/float16.hpp>

namespace ov {
namespace nvidia_gpu {
namespace kernel {

enum class Type_t : int {
    boolean,
#ifdef CUDA_HAS_BF16_TYPE
    bf16,
#endif
    f16,
    f32,
    f64,
    i4,
    i8,
    i16,
    i32,
    i64,
    u1,
    u4,
    u8,
    u16,
    u32,
    u64
};

constexpr int type_t_first_value = static_cast<int>(Type_t::boolean);
constexpr int type_t_last_value = static_cast<int>(Type_t::u64);

template <Type_t>
struct cuda_type_traits {
    using value_type = void;
};

template <Type_t Type>
using cuda_type_traits_t = typename cuda_type_traits<Type>::value_type;

template <>
struct cuda_type_traits<Type_t::boolean> {
    using value_type = char;
};

#ifdef __CUDACC__
#ifdef CUDA_HAS_BF16_TYPE
template <>
struct cuda_type_traits<Type_t::bf16> {
    using value_type = __nv_bfloat16;  // 8bit exponent, 7bit mantissa
};
#endif

template <>
struct cuda_type_traits<Type_t::f16> {
    using value_type = __half;  // 1 sign bit, 5 exponent bits, and 10 mantissa bits.
};
#endif

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
}  // namespace nvidia_gpu
}  // namespace ov
