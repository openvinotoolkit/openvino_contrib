// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <vulkan/vulkan.h>

#include "runtime/memory_manager.hpp"

namespace ov {
namespace gfx_plugin {

class VulkanGpuAllocator final : public IGpuAllocator {
public:
    explicit VulkanGpuAllocator(VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    GpuBackend backend() const override { return GpuBackend::Vulkan; }
    GpuBuffer allocate(const GpuBufferDesc& desc) override;
    GpuBuffer wrap_shared(void* ptr, size_t bytes, ov::element::Type type) override;
    void release(GpuBuffer&& buf) override;
    GpuBuffer ensure_handle(BufferHandle& handle, const GpuBufferDesc& desc);

private:
    VkBufferUsageFlags m_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
};

}  // namespace gfx_plugin
}  // namespace ov
