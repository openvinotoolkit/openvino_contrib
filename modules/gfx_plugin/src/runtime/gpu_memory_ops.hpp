// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

struct GpuBufferCopyRegion {
    size_t src_offset = 0;
    size_t dst_offset = 0;
    size_t bytes = 0;
};

struct GpuMemoryOps {
    using MapFn = void* (*)(const GpuBuffer&);
    using UnmapFn = void (*)(const GpuBuffer&);
    using FlushFn = void (*)(const GpuBuffer&, size_t, size_t);
    using InvalidateFn = void (*)(const GpuBuffer&, size_t, size_t);
    using CopyFn = void (*)(GpuCommandQueueHandle, const GpuBuffer&, const GpuBuffer&, size_t);
    using CopyRegionsFn =
        void (*)(GpuCommandQueueHandle, const GpuBuffer&, const GpuBuffer&, const GpuBufferCopyRegion*, size_t);

    MapFn map = nullptr;
    UnmapFn unmap = nullptr;
    FlushFn flush = nullptr;
    InvalidateFn invalidate = nullptr;
    CopyFn copy = nullptr;
    CopyRegionsFn copy_regions = nullptr;
};

using GpuMemoryOpsFn = const GpuMemoryOps& (*)();

bool register_memory_ops(GpuBackend backend, GpuMemoryOpsFn fn);
const GpuMemoryOps& memory_ops_for_backend(GpuBackend backend);

}  // namespace gfx_plugin
}  // namespace ov
