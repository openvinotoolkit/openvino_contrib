// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "plugin/backend_state.hpp"
#include "runtime/gpu_buffer.hpp"
#include "runtime/gpu_buffer_manager.hpp"
#include "runtime/gfx_stage_factory.hpp"

namespace ov {
namespace gfx_plugin {

struct VulkanBackendState final : BackendState {
    GpuDeviceHandle device = nullptr;
    GpuCommandQueueHandle queue = nullptr;
    GpuBufferManager* const_manager = nullptr;

    GpuBackend backend() const override { return GpuBackend::Vulkan; }
    BackendResources resources() const override {
        return {device, queue, const_manager};
    }
    bool requires_const_manager() const override { return false; }
    bool has_const_manager() const override { return const_manager != nullptr; }
    void init_infer_state(InferRequestState& state) const override;
    std::unique_ptr<GfxProfiler> create_profiler(const GfxProfilerConfig& cfg) const override;
    ov::Any get_mem_stats() const override { return {}; }
    void set_mem_stats(const ov::Any& /*stats*/) const override {}
};

}  // namespace gfx_plugin
}  // namespace ov
