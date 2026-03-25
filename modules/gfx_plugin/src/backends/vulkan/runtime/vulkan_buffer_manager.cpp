// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/vulkan_buffer_manager.hpp"

#include "backends/vulkan/runtime/vulkan_backend.hpp"
#include "openvino/core/except.hpp"
#include "runtime/memory_manager.hpp"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace ov {
namespace gfx_plugin {

namespace {

constexpr size_t kDirectMappedConstBytes = 4096;

class VulkanConstBufferReuseContext {
public:
    VulkanConstBufferReuseContext()
        : m_device_alloc(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT),
          m_staging_alloc(VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
          m_const_cache(std::make_shared<ImmutableGpuBufferCache>([this](GpuBuffer&& buf) {
              m_device_alloc.release(std::move(buf));
          })) {}

    GpuBuffer wrap_const(const std::string& key,
                         const void* data,
                         size_t bytes,
                         ov::element::Type type) {
        if (bytes == 0) {
            return {};
        }
        OPENVINO_ASSERT(data, "GFX Vulkan: const buffer data is null");
        return m_const_cache->get_or_create(key, data, bytes, type, [&]() {
            GpuBuffer staging;
            GpuBufferDesc device_desc{};
            device_desc.bytes = bytes;
            device_desc.type = type;
            device_desc.usage = BufferUsage::Const;
            device_desc.cpu_read = false;
            device_desc.cpu_write = false;
            device_desc.prefer_device_local = true;

            GpuBuffer device_buf;
            if (bytes <= kDirectMappedConstBytes) {
                // Small immutable metadata buffers are cheaper and more robust as direct
                // host-visible storage buffers than as one-off staging uploads.
                device_desc.cpu_write = true;
                device_desc.prefer_device_local = false;
                device_buf = m_device_alloc.allocate(device_desc);
                if (device_buf.valid()) {
                    gpu_copy_from_host(device_buf, data, bytes);
                }
                return device_buf;
            }

            GpuBufferDesc staging_desc{};
            staging_desc.bytes = bytes;
            staging_desc.type = type;
            staging_desc.usage = BufferUsage::Staging;
            staging_desc.cpu_write = true;
            staging_desc.prefer_device_local = false;
            staging = m_staging_alloc.allocate(staging_desc);
            if (bytes) {
                gpu_copy_from_host(staging, data, bytes);
            }
            device_buf = m_device_alloc.allocate(device_desc);
            if (bytes && staging.valid() && device_buf.valid()) {
                gpu_copy_buffer(nullptr, staging, device_buf, bytes);
            }

            if (staging.valid()) {
                m_staging_alloc.release(std::move(staging));
            }
            return device_buf;
        });
    }

private:
    VulkanGpuAllocator m_device_alloc;
    VulkanGpuAllocator m_staging_alloc;
    std::shared_ptr<ImmutableGpuBufferCache> m_const_cache;
};

class VulkanConstBufferReuseRegistry {
public:
    static VulkanConstBufferReuseRegistry& instance() {
        static VulkanConstBufferReuseRegistry registry;
        return registry;
    }

    std::shared_ptr<VulkanConstBufferReuseContext> acquire(VkDevice device) {
        const uintptr_t key = reinterpret_cast<uintptr_t>(device);
        std::lock_guard<std::mutex> lock(m_mutex);
        if (auto it = m_entries.find(key); it != m_entries.end()) {
            if (auto cached = it->second.lock()) {
                return cached;
            }
            m_entries.erase(it);
        }

        auto created = std::make_shared<VulkanConstBufferReuseContext>();
        m_entries[key] = created;
        return created;
    }

private:
    std::mutex m_mutex;
    std::unordered_map<uintptr_t, std::weak_ptr<VulkanConstBufferReuseContext>> m_entries;
};

}  // namespace

VulkanBufferManager::VulkanBufferManager()
    : m_reuse_context(VulkanConstBufferReuseRegistry::instance().acquire(VulkanContext::instance().device())) {}

VulkanBufferManager::~VulkanBufferManager() = default;

GpuBuffer VulkanBufferManager::wrap_const(const std::string& key,
                                          const void* data,
                                          size_t bytes,
                                          ov::element::Type type) {
    auto context = std::static_pointer_cast<VulkanConstBufferReuseContext>(m_reuse_context);
    OPENVINO_ASSERT(context, "GFX Vulkan: missing const buffer reuse context");
    return context->wrap_const(key, data, bytes, type);
}

const void* VulkanBufferManager::shared_const_cache_identity() const {
    return m_reuse_context.get();
}

}  // namespace gfx_plugin
}  // namespace ov
