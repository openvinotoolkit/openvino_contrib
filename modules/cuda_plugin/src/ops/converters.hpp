// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/type/element_type.hpp>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <details/ie_exception.hpp>
#include <fmt/format.h>

#include "transformer/nodes/cuda_plugin_custom_node_types.hpp"

namespace CUDAPlugin {

 /**
  * Converts OpenVINO data type to T
  * @tparam T Data type to convert
  * @param type OpenVINO data type
  * @return Converted T data type
  */
template <typename T>
T convertDataType(const ngraph::element::Type &type);

/**
 * @brief Converts OpenVINO data type to cuda data type
 */
template <>
inline constexpr cudaDataType_t convertDataType<cudaDataType_t>(const ngraph::element::Type &type) {
    using ngraph::element::Type_t;
    switch (static_cast<Type_t>(type)) {
        case Type_t::bf16: return CUDA_R_16BF;
        case Type_t::f16: return CUDA_R_16F;
        case Type_t::f32: return CUDA_R_32F;
        case Type_t::f64: return CUDA_R_64F;
        case Type_t::i8: return CUDA_R_8I;
        case Type_t::u8: return CUDA_R_8U;
        case Type_t::i16: return CUDA_R_16I;
        case Type_t::u16: return CUDA_R_16U;
        case Type_t::i32: return CUDA_R_32I;
        case Type_t::u32: return CUDA_R_32U;
        case Type_t::i64: return CUDA_R_64I;
        case Type_t::u64: return CUDA_R_64U;
        default: THROW_IE_EXCEPTION << fmt::format("The ngraph element type {} is not supported by the cuda library", type.c_type_string());
    }
}

/**
 * @brief Converts OpenVINO data type to cuDNN data type
 */
template <>
inline constexpr cudnnDataType_t convertDataType<cudnnDataType_t>(const ngraph::element::Type& type) {
    using ngraph::element::Type_t;
    switch (static_cast<Type_t>(type)) {
        case Type_t::bf16: return CUDNN_DATA_BFLOAT16;
        case Type_t::f16: return CUDNN_DATA_HALF;
        case Type_t::f32: return CUDNN_DATA_FLOAT;
        case Type_t::f64: return CUDNN_DATA_DOUBLE;
        case Type_t::i8: return CUDNN_DATA_INT8;
        case Type_t::i32: return CUDNN_DATA_INT32;
        case Type_t::i64: return CUDNN_DATA_INT64;
        default: THROW_IE_EXCEPTION << fmt::format("The ngraph element type {} is not supported by the cuDNN library", type.c_type_string());
    }
}

/**
 * @brief Converts cuda plugin activation mode to cuDNN activation mode
 */
inline constexpr cudnnActivationMode_t convertActivationMode(const nodes::ActivationMode& mode) {
    switch (mode) {
        case nodes::ActivationMode::SIGMOID: return CUDNN_ACTIVATION_SIGMOID;
        case nodes::ActivationMode::RELU: return CUDNN_ACTIVATION_RELU;
        case nodes::ActivationMode::TANH: return CUDNN_ACTIVATION_TANH;
        case nodes::ActivationMode::CLIPPED_RELU: return CUDNN_ACTIVATION_CLIPPED_RELU;
        case nodes::ActivationMode::ELU: return CUDNN_ACTIVATION_ELU;
        case nodes::ActivationMode::NO_ACTIVATION: return CUDNN_ACTIVATION_IDENTITY;
        default: THROW_IE_EXCEPTION << fmt::format("Unsupported activation: {}", mode);
    }
}

} // namespace CUDAPlugin
