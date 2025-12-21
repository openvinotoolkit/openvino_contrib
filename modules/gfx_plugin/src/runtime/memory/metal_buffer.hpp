// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>

#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {

// Keep Metal handle types ABI-stable across ObjC++ and C++ translation units.
// Always use void* and cast at call sites when ObjC types are available.
using MetalDeviceHandle = void*;
using MetalBufferHandle = void*;
using MetalHeapHandle = void*;
using MetalCommandQueueHandle = void*;
using MetalCommandBufferHandle = void*;
using MetalCommandEncoderHandle = void*;

enum class MetalStorage { Shared, Private };

enum class BufferUsage { IO, Const, Intermediate, Temp, Staging };

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

struct MetalBuffer {
    MetalBufferHandle buffer = nullptr;
    size_t size = 0;                // bytes (aligned)
    ov::element::Type type = ov::element::dynamic;
    MetalHeapHandle heap = nullptr;
    size_t offset = 0;
    bool persistent = false;
    bool from_handle = false;
    bool external = false;

    uint32_t storage_mode = 0;  // MTLStorageMode stored as integer
    uint32_t options_mask = 0;  // MTLResourceOptions mask

    bool valid() const { return buffer != nullptr; }
};

struct BufferHandle {
    MetalBuffer buf;
    size_t capacity = 0;

    bool valid() const { return buf.valid(); }
    size_t capacity_bytes() const { return capacity; }
};

}  // namespace gfx_plugin
}  // namespace ov
