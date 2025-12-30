// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "runtime/gpu_buffer.hpp"
#include "runtime/gpu_buffer_manager.hpp"

namespace ov {
namespace gfx_plugin {

struct BackendState;

struct BackendResources {
    GpuDeviceHandle device = nullptr;
    GpuCommandQueueHandle queue = nullptr;
    GpuBufferManager* const_manager = nullptr;
};

BackendResources get_backend_resources(GpuBackend backend, BackendState* state);
bool backend_has_const_manager(GpuBackend backend, BackendState* state);

}  // namespace gfx_plugin
}  // namespace ov
