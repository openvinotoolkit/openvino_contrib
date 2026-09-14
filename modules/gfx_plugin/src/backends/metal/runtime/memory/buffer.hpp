// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>

#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

// Keep Metal handle types ABI-stable across ObjC++ and C++ translation units.
// Always use void* and cast at call sites when ObjC types are available.
using MetalDeviceHandle = GpuDeviceHandle;
using MetalBufferHandle = GpuBufferHandle;
using MetalHeapHandle = GpuHeapHandle;
using MetalCommandQueueHandle = GpuCommandQueueHandle;
using MetalCommandBufferHandle = GpuCommandBufferHandle;
using MetalCommandEncoderHandle = GpuCommandEncoderHandle;

enum class MetalStorage { Shared, Private };

struct BufferDesc {
    size_t bytes = 0;
    MetalStorage storage = MetalStorage::Private;
    BufferUsage usage = BufferUsage::Intermediate;

    bool cpu_read = false;
    bool cpu_write = false;
    bool write_combined = false;

    const char* label = nullptr;
    ov::element::Type type = ov::element::dynamic;  // optional, for debugging/tracking
};

using MetalBuffer = GpuBuffer;

}  // namespace gfx_plugin
}  // namespace ov
