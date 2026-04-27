// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {

// Generic GPU handle types (backend-specific code casts these as needed).
using GpuDeviceHandle = void*;
using GpuBufferHandle = void*;
using GpuHeapHandle = void*;
using GpuCommandQueueHandle = void*;
using GpuCommandBufferHandle = void*;
using GpuCommandEncoderHandle = void*;

enum class GpuBackend { Metal, Vulkan };

enum class BufferUsage { IO, Const, Intermediate, Temp, Staging };

inline uint64_t allocate_gpu_buffer_uid() {
    static std::atomic<uint64_t> next_uid{1};
    return next_uid.fetch_add(1, std::memory_order_relaxed);
}

struct GpuBuffer {
    GpuBufferHandle buffer = nullptr;
    size_t size = 0;                // bytes (aligned)
    ov::element::Type type = ov::element::dynamic;
    GpuHeapHandle heap = nullptr;
    size_t offset = 0;
    bool persistent = false;
    bool from_handle = false;
    bool external = false;
    bool owned = true;

    uint32_t storage_mode = 0;  // backend-specific storage mode as integer
    uint32_t options_mask = 0;  // backend-specific resource options mask
    GpuBackend backend = GpuBackend::Metal;
    bool host_visible = false;
    uint64_t allocation_uid = 0;

    bool valid() const { return buffer != nullptr; }
};

struct BufferHandle {
    GpuBuffer buf;
    size_t capacity = 0;

    bool valid() const { return buf.valid(); }
    size_t capacity_bytes() const { return capacity; }
};

}  // namespace gfx_plugin
}  // namespace ov
