// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/metal_dtype.hpp"

#include <cstring>

#include "openvino/core/except.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
namespace metal_plugin {

MetalDType resolve_metal_dtype(const ov::element::Type& ov_type) {
    MetalDType dtype;
    dtype.ov_type = ov_type;

    switch (ov_type) {
        case ov::element::f16:
            dtype.compute = MetalDType::ComputeType::F32;
            dtype.storage = MetalDType::StorageType::F16;
            break;
        case ov::element::f32:
            dtype.compute = MetalDType::ComputeType::F32;
            dtype.storage = MetalDType::StorageType::F32;
            break;
        case ov::element::i32:
            dtype.compute = MetalDType::ComputeType::I32;
            dtype.storage = MetalDType::StorageType::I32;
            break;
        case ov::element::i64:
            dtype.compute = MetalDType::ComputeType::I64;
            dtype.storage = MetalDType::StorageType::I64;
            break;
        case ov::element::u8:
            dtype.compute = MetalDType::ComputeType::I32;
            dtype.storage = MetalDType::StorageType::U8;
            break;
        case ov::element::i8:
            dtype.compute = MetalDType::ComputeType::I32;
            dtype.storage = MetalDType::StorageType::I8;
            break;
        default:
            OPENVINO_THROW("METAL: unsupported element type ", ov_type.get_type_name());
    }
    return dtype;
}

size_t storage_size(const MetalDType& dtype) {
    switch (dtype.storage) {
        case MetalDType::StorageType::F16: return sizeof(ov::float16);
        case MetalDType::StorageType::F32: return sizeof(float);
        case MetalDType::StorageType::I32: return sizeof(int32_t);
        case MetalDType::StorageType::I64: return sizeof(int64_t);
        case MetalDType::StorageType::U8:  return sizeof(uint8_t);
        case MetalDType::StorageType::I8:  return sizeof(int8_t);
        default: return 0;
    }
}

size_t compute_size(const MetalDType& dtype) {
    switch (dtype.compute) {
        case MetalDType::ComputeType::F32: return sizeof(float);
        case MetalDType::ComputeType::I32: return sizeof(int32_t);
        case MetalDType::ComputeType::I64: return sizeof(int64_t);
        default: return sizeof(float);
    }
}

size_t element_size(const MetalDType& dtype) {
    return std::max(storage_size(dtype), compute_size(dtype));
}

ov::Tensor to_float32_tensor(const ov::Tensor& src) {
    const auto et = src.get_element_type();
    if (et == ov::element::f32) {
        return src;
    }

    ov::Tensor dst{ov::element::f32, src.get_shape()};
    auto* dst_data = dst.data<float>();
    const size_t count = src.get_size();

    if (et == ov::element::f16) {
        const auto* src_data = src.data<ov::float16>();
        for (size_t i = 0; i < count; ++i) dst_data[i] = static_cast<float>(src_data[i]);
        return dst;
    }
    if (et == ov::element::u8) {
        const auto* src_data = src.data<uint8_t>();
        for (size_t i = 0; i < count; ++i) dst_data[i] = static_cast<float>(src_data[i]);
        return dst;
    }
    if (et == ov::element::i8) {
        const auto* src_data = src.data<int8_t>();
        for (size_t i = 0; i < count; ++i) dst_data[i] = static_cast<float>(src_data[i]);
        return dst;
    }
    if (et == ov::element::i32) {
        const auto* src_data = src.data<int32_t>();
        for (size_t i = 0; i < count; ++i) dst_data[i] = static_cast<float>(src_data[i]);
        return dst;
    }
    if (et == ov::element::i64) {
        const auto* src_data = src.data<int64_t>();
        for (size_t i = 0; i < count; ++i) dst_data[i] = static_cast<float>(src_data[i]);
        return dst;
    }

    OPENVINO_THROW("to_float32_tensor supports only f32/f16/u8/i8/i32/i64 inputs, got ",
                   et.get_type_name());
}

}  // namespace metal_plugin
}  // namespace ov
