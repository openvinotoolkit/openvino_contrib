// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <memory>

#include "runtime/gpu_buffer_manager.hpp"
#include "runtime/immutable_gpu_buffer_cache.hpp"
#include "backends/vulkan/runtime/gpu_memory.hpp"

namespace ov {
namespace gfx_plugin {

class VulkanBufferManager final : public GpuBufferManager {
public:
    VulkanBufferManager();
    ~VulkanBufferManager() override;

    std::optional<GpuExecutionDeviceInfo> query_execution_device_info() const override;
    bool supports_const_cache() const override { return true; }
    GpuBuffer wrap_const(const std::string& key,
                         const void* data,
                         size_t bytes,
                         ov::element::Type type) override;
    const void* shared_const_cache_identity() const;

private:
    std::shared_ptr<void> m_reuse_context;
};

}  // namespace gfx_plugin
}  // namespace ov
