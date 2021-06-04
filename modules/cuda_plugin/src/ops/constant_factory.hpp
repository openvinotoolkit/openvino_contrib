// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cuda.h>
#include <cudnn.h>
#include <details/ie_exception.hpp>
#include <fmt/format.h>

namespace CUDAPlugin {

namespace constants {

union TypePlaceholder {
    std::int8_t i8;
    std::uint8_t u8;
    std::int16_t i16;
    std::uint16_t u16;
    std::int32_t i32;
    std::uint32_t u32;
    std::int64_t i64;
    std::uint64_t u64;
    __half h16;
    __nv_bfloat16 bh16;
    float f32;
    double f64;

    constexpr TypePlaceholder(std::int8_t c)
        : i8{c} {
    }
    constexpr TypePlaceholder(std::uint8_t c)
        : u8{c} {
    }
    constexpr TypePlaceholder(std::int16_t c)
        : i16{c} {
    }
    constexpr TypePlaceholder(std::uint16_t c)
        : u16{c} {
    }
    constexpr TypePlaceholder(std::int32_t c)
        : i32{c} {
    }
    constexpr TypePlaceholder(std::uint32_t c)
        : u32{c} {
    }
    constexpr TypePlaceholder(std::uint64_t c)
        : u64{c} {
    }
    constexpr TypePlaceholder(std::int64_t c)
        : i64{c} {
    }
    constexpr TypePlaceholder(__half c)
        : h16{c} {
    }
    constexpr TypePlaceholder(__nv_bfloat16 c)
        : bh16{c} {
    }
    constexpr TypePlaceholder(float c)
        : f32{c} {
    }
    constexpr TypePlaceholder(double c)
        : f64{c} {
    }
};

template<class T>
struct one {
    constexpr inline static TypePlaceholder value = static_cast<T>(1);
};

template<>
struct one<__half> {
    inline static TypePlaceholder value = __float2half(1.0f);
};

template<>
struct one<__nv_bfloat16> {
    inline static TypePlaceholder value = __float2bfloat16(1.0f);
};

template<class T>
struct zero {
    constexpr inline static TypePlaceholder value = static_cast<T>(0);
};

template<>
struct zero<__half> {
    inline static TypePlaceholder value = __float2half(0.0f);
};

template<>
struct zero<__nv_bfloat16> {
    inline static TypePlaceholder value = __float2bfloat16(0.0f);
};

} // namespace constants

/**
 * Function that returns pointer to constant
 * Usage:
 * &TypePlaceholder<constants::one>(computeType);
 * &TypePlaceholder<constants::zero>(computeType);
 * @tparam I Identifier of constant
 * @param computeType Type of constant that should returned
 * @return Pointer to constant by computeType
 */
template <template<typename T> class C>
inline const constants::TypePlaceholder& DynamicConst(cudaDataType_t computeType) {
    switch (computeType) {
        case CUDA_R_16F: {
            return C<__half>::value;
        }
        case CUDA_R_16BF: {
            return C<__nv_bfloat16>::value;
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
        case CUDA_R_16I: {
            return C<std::int16_t>::value;
        }
        case CUDA_R_16U: {
            return C<std::uint16_t>::value;
        }
        case CUDA_R_32I: {
            return C<std::int32_t>::value;
        }
        case CUDA_R_32U: {
            return C<std::uint32_t>::value;
        }
        case CUDA_R_64I: {
            return C<std::int64_t>::value;
        }
        case CUDA_R_64U: {
            return C<std::uint64_t>::value;
        }
        default: THROW_IE_EXCEPTION << fmt::format("The ngraph element type {} is not supported by the cuda library", computeType);
    }
}

template <template<typename T> class C>
inline const constants::TypePlaceholder& DynamicConst(cudnnDataType_t computeType) {
    switch (computeType) {
        case CUDNN_DATA_DOUBLE: {
            return C<double>::value;
        }
        default: {
            return C<float>::value;
        }
    }
}

} // namespace CUDAPlugin
