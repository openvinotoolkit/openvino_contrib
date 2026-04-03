// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

struct GpuTensor {
    GpuBuffer buf;
    ov::Shape shape;
    ov::element::Type expected_type = ov::element::dynamic;  // desired logical type
    bool prefer_private = true;
};

}  // namespace gfx_plugin
}  // namespace ov
