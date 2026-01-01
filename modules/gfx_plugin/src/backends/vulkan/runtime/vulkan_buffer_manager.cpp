// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/vulkan_buffer_manager.hpp"

#include "openvino/core/except.hpp"
#include "runtime/memory_manager.hpp"

namespace ov {
namespace gfx_plugin {

VulkanBufferManager::VulkanBufferManager()
    : m_device_alloc(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT),
      m_staging_alloc(VK_BUFFER_USAGE_TRANSFER_SRC_BIT) {}

VulkanBufferManager::~VulkanBufferManager() {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& kv : m_cache) {
        if (kv.second.buffer.valid()) {
            m_device_alloc.release(std::move(kv.second.buffer));
        }
    }
    m_cache.clear();
}

GpuBuffer VulkanBufferManager::wrap_const(const std::string& key,
                                          const void* data,
                                          size_t bytes,
                                          ov::element::Type type) {
    if (bytes == 0) {
        return {};
    }
    OPENVINO_ASSERT(data, "GFX Vulkan: const buffer data is null");
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_cache.find(key);
    if (it != m_cache.end() && it->second.buffer.valid()) {
        return it->second.buffer;
    }

    GpuBufferDesc staging_desc{};
    staging_desc.bytes = bytes;
    staging_desc.type = type;
    staging_desc.usage = BufferUsage::Staging;
    staging_desc.cpu_write = true;
    staging_desc.prefer_device_local = false;
    GpuBuffer staging = m_staging_alloc.allocate(staging_desc);
    if (bytes) {
        gpu_copy_from_host(staging, data, bytes);
    }

    GpuBufferDesc device_desc{};
    device_desc.bytes = bytes;
    device_desc.type = type;
    device_desc.usage = BufferUsage::Const;
    device_desc.cpu_read = false;
    device_desc.cpu_write = false;
    device_desc.prefer_device_local = true;
    GpuBuffer device_buf = m_device_alloc.allocate(device_desc);
    if (bytes && staging.valid() && device_buf.valid()) {
        gpu_copy_buffer(nullptr, staging, device_buf, bytes);
    }

    if (staging.valid()) {
        m_staging_alloc.release(std::move(staging));
    }

    ConstEntry entry{};
    entry.buffer = device_buf;
    entry.bytes = bytes;
    entry.type = type;
    m_cache.emplace(key, entry);
    return entry.buffer;
}

}  // namespace gfx_plugin
}  // namespace ov
