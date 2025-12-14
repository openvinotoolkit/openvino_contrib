// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/visibility.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace metal_plugin {

// Unified dtype descriptor for METAL backend: captures original OV type,
// compute type inside Metal kernels, and storage type for buffers.
struct MetalDType {
    enum class ComputeType { F32, I32, I64 };
    enum class StorageType { F32, F16, I32, I64, U8, I8 };

    ov::element::Type ov_type{ov::element::dynamic};
    ComputeType compute{ComputeType::F32};
    StorageType storage{StorageType::F32};

    bool is_float() const { return compute == ComputeType::F32; }
};

OPENVINO_API MetalDType resolve_metal_dtype(const ov::element::Type& ov_type);
OPENVINO_API size_t storage_size(const MetalDType& dtype);

// Debug/helper utility: produce a float32 host tensor from any float-like input.
// Returns the same tensor if already f32; otherwise allocates a new buffer.
OPENVINO_API ov::Tensor to_float32_tensor(const ov::Tensor& src);

}  // namespace metal_plugin
}  // namespace ov
