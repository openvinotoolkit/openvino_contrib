// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "backends/metal/runtime/metal_memory.hpp"
#include "plugin/backend_state.hpp"

namespace ov {
namespace gfx_plugin {

struct MetalBackendState final : BackendState {
    MetalDeviceHandle device = nullptr;
    MetalCommandQueueHandle command_queue = nullptr;
    MetalDeviceCaps caps{};
    std::unique_ptr<MetalAllocatorCore> alloc_core;
    std::unique_ptr<MetalHeapPool> persistent_heaps;
    std::unique_ptr<MetalFreeList> persistent_freelist;
    std::unique_ptr<MetalStagingPool> persistent_staging;
    std::unique_ptr<MetalAllocator> persistent_alloc;
    std::unique_ptr<MetalConstCache> const_cache;
    std::shared_ptr<MetalBufferManager> const_manager;
    MetalMemoryStats dummy_stats{};
    mutable MetalMemoryStats last_stats{};

    GpuBackend backend() const override { return GpuBackend::Metal; }
};

}  // namespace gfx_plugin
}  // namespace ov
