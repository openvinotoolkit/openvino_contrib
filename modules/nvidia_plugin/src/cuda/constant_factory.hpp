// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn_ops_infer.h>
#include <fmt/format.h>

#include <cuda/float16.hpp>
#include <error.hpp>

namespace CUDA {

namespace constants {
/**
 * AnyNumeric - a union, combining all numeric types in use
 */
union AnyNumeric {
    std::int8_t i8;
    std::uint8_t u8;
    std::int16_t i16;
    std::uint16_t u16;
    std::int32_t i32;
    std::uint32_t u32;
    std::int64_t i64;
    std::uint64_t u64;
    __half h16;
#ifdef CUDA_HAS_BF16_TYPE
    __nv_bfloat16 bh16;
    explicit constexpr AnyNumeric(__nv_bfloat16 c) : bh16{c} {}
#endif
    float f32;
    double f64;
    AnyNumeric() = delete;
    AnyNumeric(const AnyNumeric&) = delete;
    AnyNumeric(AnyNumeric&&) = delete;
    AnyNumeric& operator=(AnyNumeric&) = delete;

    explicit constexpr AnyNumeric(std::int8_t c) : i8{c} {}
    explicit constexpr AnyNumeric(std::uint8_t c) : u8{c} {}
    explicit constexpr AnyNumeric(std::int16_t c) : i16{c} {}
    explicit constexpr AnyNumeric(std::uint16_t c) : u16{c} {}
    explicit constexpr AnyNumeric(std::int32_t c) : i32{c} {}
    explicit constexpr AnyNumeric(std::uint32_t c) : u32{c} {}
    explicit constexpr AnyNumeric(std::uint64_t c) : u64{c} {}
    explicit constexpr AnyNumeric(std::int64_t c) : i64{c} {}
    explicit constexpr AnyNumeric(__half c) : h16{c} {}
    explicit constexpr AnyNumeric(float c) : f32{c} {}
    explicit constexpr AnyNumeric(double c) : f64{c} {}
};

template <class T>
struct one {
    constexpr inline static AnyNumeric value{static_cast<T>(1)};
};

template <>
struct one<__half> {
    const inline static AnyNumeric value{__float2half(1.0f)};
};

#ifdef CUDA_HAS_BF16_TYPE
template <>
struct one<__nv_bfloat16> {
    const inline static AnyNumeric value{__float2bfloat16(1.0f)};
};
#endif

template <class T>
struct minusOne {
    constexpr inline static AnyNumeric value{static_cast<T>(-1)};
};

template <>
struct minusOne<__half> {
    const inline static AnyNumeric value{__float2half(-1.0f)};
};

#ifdef CUDA_HAS_BF16_TYPE
template <>
struct minusOne<__nv_bfloat16> {
    const inline static AnyNumeric value{__float2bfloat16(-1.0f)};
};
#endif

template <class T>
struct zero {
    constexpr inline static AnyNumeric value{static_cast<T>(0)};
};

template <>
struct zero<__half> {
    const inline static AnyNumeric value{__float2half(0.0f)};
};

#ifdef CUDA_HAS_BF16_TYPE
template <>
struct zero<__nv_bfloat16> {
    const inline static AnyNumeric value{__float2bfloat16(0.0f)};
};
#endif

}  // namespace constants

/**
 * Function that returns reference to static constant
 * Usage:
 * &NumericConst<constants::one>(computeType);
 * &NumericConst<constants::zero>(computeType);
 * @tparam C a class declaring the constant as its static member value
 * @param computeType Type of constant that should returned
 * @return Reference to AnyNumeric, containing constant as dictated by computeType
 */
template <template <typename T> class C>
inline const constants::AnyNumeric& NumericConst(cudaDataType_t computeType) {
    switch (computeType) {
#ifdef CUDA_HAS_BF16_TYPE
        case CUDA_R_16BF: {
            return C<__nv_bfloat16>::value;
        }
        case CUDA_R_16I: {
            return C<std::int16_t>::value;
        }
        case CUDA_R_16U: {
            return C<std::uint16_t>::value;
        }
        case CUDA_R_64I: {
            return C<std::int64_t>::value;
        }
        case CUDA_R_64U: {
            return C<std::uint64_t>::value;
        }
#endif
        case CUDA_R_16F: {
            return C<__half>::value;
        }
        case CUDA_R_32F: {
            return C<float>::value;
        }
        case CUDA_R_64F: {
            return C<double>::value;
        }
        case CUDA_R_8I: {
            return C<std::int8_t>::value;
        }
        case CUDA_R_8U: {
            return C<std::uint8_t>::value;
        }
        case CUDA_R_32I: {
            return C<std::int32_t>::value;
        }
        case CUDA_R_32U: {
            return C<std::uint32_t>::value;
        }
        default:
            ov::nvidia_gpu::throw_ov_exception(
                fmt::format("The ngraph element type {} is not supported by "
                            "the cuda library",
                            computeType));
    }
}

template <template <typename T> class C>
inline constexpr const constants::AnyNumeric& NumericConst(cudnnDataType_t computeType) {
    switch (computeType) {
        case CUDNN_DATA_DOUBLE: {
            return C<double>::value;
        }
        default: {
            return C<float>::value;
        }
    }
}
}  // namespace CUDA
