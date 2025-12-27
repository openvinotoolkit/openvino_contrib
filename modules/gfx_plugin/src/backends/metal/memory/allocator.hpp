// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "backends/metal/memory/allocator_core.hpp"
#include "backends/metal/memory/freelist.hpp"
#include "backends/metal/memory/heap_pool.hpp"
#include "backends/metal/memory/memory_stats.hpp"
#include "backends/metal/memory/staging_pool.hpp"

namespace ov {
namespace gfx_plugin {

class MetalProfiler;

class MetalAllocator {
public:
    MetalAllocator(MetalAllocatorCore& core,
                   MetalHeapPool& heaps,
                   MetalFreeList& freelist,
                   MetalStagingPool& staging,
                   MetalDeviceCaps caps);

    MetalBuffer allocate(const BufferDesc& desc, bool persistent);
    MetalBuffer allocate_staging(size_t bytes, const char* label = nullptr);

    MetalBuffer ensure_handle(BufferHandle& handle, const BufferDesc& desc, bool persistent);

    void release(MetalBuffer&& buf);

    const MetalMemoryStats& stats() const { return m_stats; }
    void reset_stats() { m_stats = {}; }

    MetalAllocatorCore& core() { return m_core; }
    void set_profiler(MetalProfiler* profiler, bool detailed);

private:
    size_t align_size(size_t size) const;
    size_t bucket_size(size_t size) const;
    FreeKey make_key(size_t bucket, const BufferDesc& desc, uint32_t options_mask) const;

    MetalAllocatorCore& m_core;
    MetalHeapPool& m_heaps;
    MetalFreeList& m_freelist;
    MetalStagingPool& m_staging;
    MetalDeviceCaps m_caps;
    MetalMemoryStats m_stats{};
    MetalProfiler* m_profiler = nullptr;
    bool m_profile_allocations = false;
};

}  // namespace gfx_plugin
}  // namespace ov
