// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

#include "runtime/gpu_buffer_manager.hpp"
#include "backends/vulkan/runtime/gpu_memory.hpp"

namespace ov {
namespace gfx_plugin {

class VulkanBufferManager final : public GpuBufferManager {
public:
    VulkanBufferManager();
    ~VulkanBufferManager() override;

    bool supports_const_cache() const override { return true; }
    GpuBuffer wrap_const(const std::string& key,
                         const void* data,
                         size_t bytes,
                         ov::element::Type type) override;

private:
    struct ConstEntry {
        GpuBuffer buffer{};
        size_t bytes = 0;
        ov::element::Type type = ov::element::dynamic;
    };

    mutable std::mutex m_mutex;
    std::unordered_map<std::string, ConstEntry> m_cache;
    VulkanGpuAllocator m_device_alloc;
    VulkanGpuAllocator m_staging_alloc;
};

}  // namespace gfx_plugin
}  // namespace ov
