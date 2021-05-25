// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
        case Type_t::i32: return CUDA_R_32I;
        case Type_t::u32: return CUDA_R_32U;
        case Type_t::i64: return CUDA_R_64I;
        case Type_t::u64: return CUDA_R_64U;
        default:THROW_IE_EXCEPTION << "Unsupported ngraph element type " << type.c_type_string();
    }
}

} // namespace CUDAPlugin
