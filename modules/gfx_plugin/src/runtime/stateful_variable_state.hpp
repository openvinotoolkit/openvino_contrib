// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <unordered_map>

#include "openvino/core/partial_shape.hpp"
#include "openvino/runtime/tensor.hpp"

#include "runtime/gpu_buffer.hpp"
#include "runtime/gpu_tensor.hpp"

namespace ov {
namespace gfx_plugin {

struct StatefulVariableTensorState {
    GpuTensor tensor;
    BufferHandle handle;
    BufferHandle upload_handle;
    ov::PartialShape expected_shape;
    ov::element::Type expected_type = ov::element::dynamic;
    ov::Tensor host_tensor;
    bool initialized = false;
    bool host_dirty = false;
    bool host_stale = false;
};

using StatefulVariableStateMap =
    std::unordered_map<std::string, StatefulVariableTensorState>;

}  // namespace gfx_plugin
}  // namespace ov
