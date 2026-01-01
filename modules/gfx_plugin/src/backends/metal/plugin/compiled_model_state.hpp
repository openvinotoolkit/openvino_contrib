// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "backends/metal/runtime/metal_memory.hpp"
#include "plugin/backend_state.hpp"
#include "runtime/gfx_stage_factory.hpp"

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
    BackendResources resources() const override {
        return {device, command_queue, const_manager.get()};
    }
    bool requires_const_manager() const override { return true; }
    bool has_const_manager() const override { return const_manager != nullptr; }
    void release() override {
        if (command_queue) {
            metal_release_command_queue(command_queue);
            command_queue = nullptr;
        }
    }
    void init_infer_state(InferRequestState& state) const override;
    ov::SoPtr<ov::ITensor> get_tensor_override(
        const InferRequestState& state,
        size_t idx,
        const std::vector<ov::Output<const ov::Node>>& outputs) const override;
    ov::Any get_mem_stats() const override { return ov::Any{last_stats}; }
    void set_mem_stats(const ov::Any& stats) const override {
        if (stats.is<MetalMemoryStats>()) {
            last_stats = stats.as<MetalMemoryStats>();
        }
    }
    std::unique_ptr<GfxProfiler> create_profiler(const GfxProfilerConfig& cfg) const override;
};

}  // namespace gfx_plugin
}  // namespace ov
