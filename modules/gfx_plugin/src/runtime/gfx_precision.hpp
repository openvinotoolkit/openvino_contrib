// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {

inline ov::element::Type gfx_default_inference_precision() {
    return ov::element::f16;
}

inline bool gfx_uses_fp16_compute(const ov::element::Type& type) {
    return type == ov::element::f16 || type == ov::element::f32 || type == ov::element::bf16;
}

inline ov::element::Type gfx_compute_element_type(const ov::element::Type& storage_type) {
    return gfx_uses_fp16_compute(storage_type) ? ov::element::f16 : storage_type;
}

inline ov::element::Type gfx_runtime_element_type(const ov::element::Type& type) {
    return gfx_compute_element_type(type);
}

}  // namespace gfx_plugin
}  // namespace ov
