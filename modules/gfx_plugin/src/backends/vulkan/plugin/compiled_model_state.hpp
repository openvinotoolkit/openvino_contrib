// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "plugin/backend_state.hpp"
#include "runtime/gpu_buffer.hpp"
#include "runtime/gpu_buffer_manager.hpp"

namespace ov {
namespace gfx_plugin {

struct VulkanBackendState final : BackendState {
    GpuDeviceHandle device = nullptr;
    GpuCommandQueueHandle queue = nullptr;
    GpuBufferManager* const_manager = nullptr;

    GpuBackend backend() const override { return GpuBackend::Vulkan; }
};

}  // namespace gfx_plugin
}  // namespace ov
