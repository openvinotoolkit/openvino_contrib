// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "backends/metal/runtime/metal_memory.hpp"
#include "backends/metal/runtime/stage_factory.hpp"
#include "runtime/backend_runtime.hpp"

namespace ov {
namespace gfx_plugin {

struct MetalBackendState final : BackendState {
    compiler::BackendTarget runtime_target;
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

    const compiler::BackendTarget& target() const override { return runtime_target; }
    GpuBackend backend() const override { return runtime_target.backend(); }
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
    void init_infer_state(BackendRequestState& state) const override;
    std::unique_ptr<GpuStage> create_stage(
        const RuntimeStageMaterializationContext& context) const override {
        return create_metal_stage(context, device, command_queue);
    }
    ov::SoPtr<ov::ITensor> get_tensor_override(
        const BackendRequestState& state,
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
