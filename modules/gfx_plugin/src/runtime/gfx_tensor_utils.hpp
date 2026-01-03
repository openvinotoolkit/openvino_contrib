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

}  // namespace gfx_plugin
}  // namespace ov
