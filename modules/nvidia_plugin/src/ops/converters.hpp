// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn.h>
#include <fmt/format.h>

#include <cstdint>
#include <cuda/float16.hpp>
#include <error.hpp>
#include <kernels/details/cuda_type_traits.hpp>
#include <openvino/core/except.hpp>
#include <string_view>
#include <type_traits>

#include "transformer/nodes/activation_type.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * Converts OpenVINO data type to T
 * @tparam T Data type to convert
 * @param type OpenVINO data type
 * @return Converted T data type
 */
template <typename T>
T convertDataType(ov::element::Type type);

/**
 * @brief Converts OpenVINO data type to cuda data type
 */
template <>
inline constexpr cudaDataType_t convertDataType<cudaDataType_t>(const ov::element::Type type) {
    using ov::element::Type_t;
    switch (static_cast<Type_t>(type)) {
#if CUDART_VERSION >= 11000
        case Type_t::bf16:
            return CUDA_R_16BF;
        case Type_t::i16:
            return CUDA_R_16I;
        case Type_t::u16:
            return CUDA_R_16U;
        case Type_t::i64:
            return CUDA_R_64I;
        case Type_t::u64:
            return CUDA_R_64U;
#endif
        case Type_t::f16:
            return CUDA_R_16F;
        case Type_t::f32:
            return CUDA_R_32F;
        case Type_t::f64:
            return CUDA_R_64F;
        case Type_t::i8:
            return CUDA_R_8I;
        case Type_t::u8:
            return CUDA_R_8U;
        case Type_t::i32:
            return CUDA_R_32I;
        case Type_t::u32:
            return CUDA_R_32U;
        default:
            throw_ov_exception(
                fmt::format("The ngraph element type {} is not supported by "
                            "the cuda library",
                            type.c_type_string()));
    }
}

/**
 * @brief Converts OpenVINO data type to cuDNN data type
 */
template <>
inline constexpr cudnnDataType_t convertDataType<cudnnDataType_t>(const ov::element::Type type) {
    using ov::element::Type_t;
    switch (static_cast<Type_t>(type)) {
        case Type_t::boolean:
            return CUDNN_DATA_HALF;
        case Type_t::bf16:
            return CUDNN_DATA_BFLOAT16;
        case Type_t::f16:
            return CUDNN_DATA_HALF;
        case Type_t::f32:
            return CUDNN_DATA_FLOAT;
        case Type_t::f64:
            return CUDNN_DATA_DOUBLE;
        case Type_t::i8:
            return CUDNN_DATA_INT8;
        case Type_t::i32:
            return CUDNN_DATA_INT32;
        case Type_t::i64:
            return CUDNN_DATA_INT64;
        default:
            throw_ov_exception(
                fmt::format("The ngraph element type {} is not supported by "
                            "the cuDNN library",
                            type.c_type_string()));
    }
}

template <>
inline constexpr kernel::Type_t convertDataType<kernel::Type_t>(const ov::element::Type type) {
    using nType_t = ov::element::Type_t;
    using kType_t = kernel::Type_t;
    switch (static_cast<nType_t>(type)) {
        case nType_t::boolean:
            return kType_t::boolean;
#ifdef CUDA_HAS_BF16_TYPE
        case nType_t::bf16:
            return kType_t::bf16;
#endif
        case nType_t::i16:
            return kType_t::i16;
        case nType_t::u16:
            return kType_t::u16;
        case nType_t::i64:
            return kType_t::i64;
        case nType_t::u64:
            return kType_t::u64;
        case nType_t::f16:
            return kType_t::f16;
        case nType_t::f32:
            return kType_t::f32;
        case nType_t::f64:
            return kType_t::f64;
        case nType_t::u1:
            return kType_t::u1;
        case nType_t::i4:
            return kType_t::i4;
        case nType_t::u4:
            return kType_t::u4;
        case nType_t::i8:
            return kType_t::i8;
        case nType_t::u8:
            return kType_t::u8;
        case nType_t::i32:
            return kType_t::i32;
        case nType_t::u32:
            return kType_t::u32;
        default:
            throw_ov_exception(
                fmt::format("The ngraph element type {} is not supported by "
                            "the cuda library",
                            type.c_type_string()));
    }
}

/**
 * @brief Retruns std::string representation of T type
 */
template <typename T>
std::string_view toString(const T& type);

/**
 * @brief Retruns std::string representation of cudaDataType_t type
 */
template <>
inline constexpr std::string_view toString<cudaDataType_t>(const cudaDataType_t& type) {
    switch (type) {
#if CUDART_VERSION >= 11000
        case CUDA_R_16BF:
            return "CUDA_R_16BF";
        case CUDA_R_16I:
            return "CUDA_R_16I";
        case CUDA_R_16U:
            return "CUDA_R_16U";
        case CUDA_R_64I:
            return "CUDA_R_64I";
        case CUDA_R_64U:
            return "CUDA_R_64U";
#endif
        case CUDA_R_16F:
            return "CUDA_R_16F";
        case CUDA_R_32F:
            return "CUDA_R_32F";
        case CUDA_R_64F:
            return "CUDA_R_64F";
        case CUDA_R_8I:
            return "CUDA_R_8I";
        case CUDA_R_8U:
            return "CUDA_R_8U";
        case CUDA_R_32I:
            return "CUDA_R_32I";
        case CUDA_R_32U:
            return "CUDA_R_32U";
        default:
            throw_ov_exception(
                fmt::format("ov::nvidia_gpu::toString<cudaDataType_t>(): Unsupported data type: type = {}", type));
    }
}

/**
 * @brief Retruns std::string representation of cudnnDataType_t type
 */
template <>
inline constexpr std::string_view toString<cudnnDataType_t>(const cudnnDataType_t& type) {
    switch (type) {
        case CUDNN_DATA_FLOAT:
            return "CUDNN_DATA_FLOAT";
        case CUDNN_DATA_DOUBLE:
            return "CUDNN_DATA_DOUBLE";
        case CUDNN_DATA_HALF:
            return "CUDNN_DATA_HALF";
        case CUDNN_DATA_INT8:
            return "CUDNN_DATA_INT8";
        case CUDNN_DATA_INT32:
            return "CUDNN_DATA_INT32";
        case CUDNN_DATA_INT8x4:
            return "CUDNN_DATA_INT8x4";
        case CUDNN_DATA_UINT8:
            return "CUDNN_DATA_UINT8";
        case CUDNN_DATA_UINT8x4:
            return "CUDNN_DATA_UINT8x4";
        case CUDNN_DATA_INT8x32:
            return "CUDNN_DATA_INT8x32";
        case CUDNN_DATA_BFLOAT16:
            return "CUDNN_DATA_BFLOAT16";
        case CUDNN_DATA_INT64:
            return "CUDNN_DATA_INT64";
        default:
            throw_ov_exception(
                fmt::format("ov::nvidia_gpu::toString<cudaDataType_t>(): Unsupported data type: type = {}", type));
    }
}

/**
 * @brief Retruns the size of cudnnDataType_t type value in bytes
 */
inline constexpr std::size_t elementSize(cudnnDataType_t type) {
    switch (type) {
        case CUDNN_DATA_FLOAT:
            return sizeof(float);
        case CUDNN_DATA_DOUBLE:
            return sizeof(double);
        case CUDNN_DATA_HALF:
            return sizeof(float) / 2;
        case CUDNN_DATA_INT8:
            return sizeof(std::int8_t);
        case CUDNN_DATA_INT32:
            return sizeof(std::int32_t);
        case CUDNN_DATA_INT8x4:
            return sizeof(std::int8_t) * 4;
        case CUDNN_DATA_UINT8:
            return sizeof(std::uint8_t);
        case CUDNN_DATA_UINT8x4:
            return sizeof(std::uint8_t) * 4;
        case CUDNN_DATA_INT8x32:
            return sizeof(std::int8_t) * 32;
        case CUDNN_DATA_BFLOAT16:
            return sizeof(std::uint16_t);
        case CUDNN_DATA_INT64:
            return sizeof(std::int64_t);
        default:
            throw_ov_exception(
                fmt::format("The cudnnDataType_t {} is not supported by the cuDNN library", toString(type)));
    }
}

/**
 * @brief Converts cuda plugin activation mode to cuDNN Backend API activation mode
 */
inline constexpr cudnnPointwiseMode_t convertActivationModeToBE(const nodes::ActivationMode& mode) {
    switch (mode) {
        case nodes::ActivationMode::SIGMOID:
            return CUDNN_POINTWISE_SIGMOID_FWD;
        case nodes::ActivationMode::RELU:
            return CUDNN_POINTWISE_RELU_FWD;
        case nodes::ActivationMode::TANH:
            return CUDNN_POINTWISE_TANH_FWD;
        case nodes::ActivationMode::ELU:
            return CUDNN_POINTWISE_GELU_FWD;
        case nodes::ActivationMode::SWISH:
            return CUDNN_POINTWISE_SWISH_FWD;
        default:
            throw_ov_exception(fmt::format("Unsupported activation: {}", mode));
    }
}

inline constexpr cudnnActivationMode_t convertActivationMode(const nodes::ActivationMode& mode) {
    switch (mode) {
        case nodes::ActivationMode::SIGMOID:
            return CUDNN_ACTIVATION_SIGMOID;
        case nodes::ActivationMode::RELU:
            return CUDNN_ACTIVATION_RELU;
        case nodes::ActivationMode::TANH:
            return CUDNN_ACTIVATION_TANH;
        case nodes::ActivationMode::CLIPPED_RELU:
            return CUDNN_ACTIVATION_CLIPPED_RELU;
        case nodes::ActivationMode::ELU:
            return CUDNN_ACTIVATION_ELU;
        case nodes::ActivationMode::NO_ACTIVATION:
            return CUDNN_ACTIVATION_IDENTITY;
        default:
            throw_ov_exception(fmt::format("Unsupported activation: {}", mode));
    }
}

/**
 * @brief Auxilary structure for E type used in SwitchCase()
 * should contain type alias and static constexpr shift member of unsigned type, see example below
 */
template <typename E>
struct SwitchCaseTrait;

/**
 * @brief Auxilary structure for cudnnDataType_t used in SwitchCase()
 */
template <>
struct SwitchCaseTrait<cudnnDataType_t> {
    using type = uint32_t;
    static constexpr std::size_t shift = 16;
};

/**
 * @brief Packs two enum or integer values into one integer value allowing it to be used in switch() and case statements
 * e.g. switch (switchCase(in0, out)) {
 *          case switchCase(CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT):
 * Before using this function SwitchCaseTrait<E> structure specification has to be defined for E type, see example adove
 */
template <typename E>
inline constexpr typename SwitchCaseTrait<E>::type switchCase(E first, E second) {
    using I = typename SwitchCaseTrait<E>::type;
    static_assert(std::is_enum<E>() || std::is_integral<E>());
    static_assert(std::is_integral<I>());
    static_assert(std::is_unsigned<decltype(SwitchCaseTrait<E>::shift)>());
    constexpr auto shift = SwitchCaseTrait<E>::shift;
    const I firstI = static_cast<I>(first);
    const I secondI = static_cast<I>(second);
    const I result = (firstI << shift) + secondI;
    OPENVINO_ASSERT(static_cast<E>(secondI) == second);
    OPENVINO_ASSERT(static_cast<E>((result - secondI) >> shift) == first);
    return result;
}

// TODO: use in CuDnnTensorOpBase
/**
 * @brief Gets opTensorCompType for opTensorDesc used in cudnnOpTensor()
 * See https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnOpTensor for more details.
 */
inline constexpr cudnnDataType_t getCuDnnOpTensorCompType(cudnnDataType_t in0,
                                                          cudnnDataType_t in1,
                                                          cudnnDataType_t out) {
    auto throwException = [=] {
        throw_ov_exception(
            fmt::format("ov::nvidia_gpu::getCuDnnOpTensorType(): Unsupported data types: in0 = {}, in1 = {} out = {}",
                        toString(in0),
                        toString(in1),
                        toString(out)));
    };
    if (in0 != in1) {
        throwException();
    }
    switch (switchCase(in0, out)) {
        case switchCase(CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT):
        case switchCase(CUDNN_DATA_INT8, CUDNN_DATA_FLOAT):
        case switchCase(CUDNN_DATA_HALF, CUDNN_DATA_FLOAT):
        case switchCase(CUDNN_DATA_BFLOAT16, CUDNN_DATA_FLOAT):
        case switchCase(CUDNN_DATA_FLOAT, CUDNN_DATA_HALF):
        case switchCase(CUDNN_DATA_HALF, CUDNN_DATA_HALF):
        case switchCase(CUDNN_DATA_INT8, CUDNN_DATA_INT8):
        case switchCase(CUDNN_DATA_FLOAT, CUDNN_DATA_INT8):
        case switchCase(CUDNN_DATA_FLOAT, CUDNN_DATA_BFLOAT16):
        case switchCase(CUDNN_DATA_BFLOAT16, CUDNN_DATA_BFLOAT16):
            return CUDNN_DATA_FLOAT;
        case switchCase(CUDNN_DATA_DOUBLE, CUDNN_DATA_DOUBLE):
            return CUDNN_DATA_DOUBLE;
        default:
            throwException();
    }
    return CUDNN_DATA_FLOAT;  // never reached
}

}  // namespace nvidia_gpu
}  // namespace ov
