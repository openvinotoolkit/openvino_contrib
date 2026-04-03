// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "backends/vulkan/runtime/stage_factory.hpp"
#include "plugin/backend_state.hpp"
#include "runtime/gpu_buffer.hpp"
#include "runtime/gpu_buffer_manager.hpp"

namespace ov {
namespace gfx_plugin {

struct VulkanBackendState final : BackendState {
    GpuDeviceHandle device = nullptr;
    GpuCommandQueueHandle queue = nullptr;
    std::shared_ptr<GpuBufferManager> const_manager;

    GpuBackend backend() const override { return GpuBackend::Vulkan; }
    BackendResources resources() const override {
        return {device, queue, const_manager.get()};
    }
    bool requires_const_manager() const override { return true; }
    bool has_const_manager() const override { return const_manager != nullptr; }
    void init_infer_state(InferRequestState& state) const override;
    std::unique_ptr<GpuStage> create_stage(const std::shared_ptr<const ov::Node>& node) const override {
        return create_vulkan_stage(node, device, queue);
    }
    std::unique_ptr<GfxProfiler> create_profiler(const GfxProfilerConfig& cfg) const override;
    ov::Any get_mem_stats() const override { return {}; }
    void set_mem_stats(const ov::Any& /*stats*/) const override {}
};

}  // namespace gfx_plugin
}  // namespace ov
