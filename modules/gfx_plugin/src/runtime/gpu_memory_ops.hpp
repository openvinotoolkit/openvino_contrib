// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

struct GpuMemoryOps {
    using MapFn = void* (*)(const GpuBuffer&);
    using UnmapFn = void (*)(const GpuBuffer&);
    using FlushFn = void (*)(const GpuBuffer&, size_t, size_t);
    using InvalidateFn = void (*)(const GpuBuffer&, size_t, size_t);
    using CopyFn = void (*)(GpuCommandQueueHandle, const GpuBuffer&, const GpuBuffer&, size_t);

    MapFn map = nullptr;
    UnmapFn unmap = nullptr;
    FlushFn flush = nullptr;
    InvalidateFn invalidate = nullptr;
    CopyFn copy = nullptr;
};

const GpuMemoryOps& memory_ops_for_backend(GpuBackend backend);

// Implemented in backend-specific runtime units (or stubs when backend is unavailable).
const GpuMemoryOps& metal_memory_ops();
const GpuMemoryOps& vulkan_memory_ops();

}  // namespace gfx_plugin
}  // namespace ov
