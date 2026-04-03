// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/visibility.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace gfx_plugin {

// Debug/helper utility: produce a float32 host tensor from any float-like input.
// Returns the same tensor if already f32; otherwise allocates a new buffer.
OPENVINO_API ov::Tensor to_float32_tensor(const ov::Tensor& src);

OPENVINO_API ov::Tensor convert_tensor_element_type(const ov::Tensor& src,
                                                    const ov::element::Type& dst_type);

}  // namespace gfx_plugin
}  // namespace ov
