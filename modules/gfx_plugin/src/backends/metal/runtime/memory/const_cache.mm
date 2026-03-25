// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/memory/const_cache.hpp"

#include "backends/metal/runtime/metal_memory.hpp"
#include "openvino/core/except.hpp"

#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace ov {
namespace gfx_plugin {

class MetalConstCacheContext {
public:
    explicit MetalConstCacheContext(MetalDeviceHandle device)
        : m_device(device),
          m_queue(metal_create_command_queue(device)),
          m_core(std::make_unique<MetalAllocatorCore>(device, query_metal_device_caps(device))),
          m_cache(std::make_shared<ImmutableGpuBufferCache>([this](GpuBuffer&& buf) {
              MetalBuffer metal = buf;
              m_core->release_buffer(metal);
          })) {
        OPENVINO_ASSERT(m_device, "GFX Metal: const cache device is null");
        OPENVINO_ASSERT(m_queue, "GFX Metal: failed to create const cache command queue");
    }

    ~MetalConstCacheContext() {
        m_cache.reset();
        if (m_queue) {
            metal_release_command_queue(m_queue);
            m_queue = nullptr;
        }
    }

    std::shared_ptr<ImmutableGpuBufferCache> cache() const {
        return m_cache;
    }

    MetalBuffer create_constant_buffer(const void* data, size_t bytes, const BufferDesc& desc) {
        MetalBuffer buf;
        if (desc.storage == MetalStorage::Shared) {
            buf = m_core->wrap_shared(const_cast<void*>(data), bytes, desc.type);
            buf.persistent = true;
            buf.external = true;
            buf.from_handle = true;
            return buf;
        }

        BufferDesc local = desc;
        local.storage = MetalStorage::Private;
        buf = m_core->create_buffer(local);

        BufferDesc staging_desc;
        staging_desc.bytes = bytes;
        staging_desc.type = desc.type;
        staging_desc.storage = MetalStorage::Shared;
        staging_desc.usage = BufferUsage::Staging;
        staging_desc.write_combined = true;

        MetalBuffer staging = m_core->create_buffer(staging_desc);
        OPENVINO_ASSERT(staging.valid(), "GFX: failed to allocate Metal staging buffer for constants");
        if (bytes) {
            void* mapped = metal_map_buffer(staging);
            OPENVINO_ASSERT(mapped, "GFX: failed to map Metal staging buffer for constants");
            std::memcpy(mapped, data, bytes);
            metal_unmap_buffer(staging);
            metal_copy_buffer(m_queue, staging, buf, bytes);
        }
        m_core->release_buffer(staging);
        return buf;
    }

private:
    MetalDeviceHandle m_device = nullptr;
    MetalCommandQueueHandle m_queue = nullptr;
    std::unique_ptr<MetalAllocatorCore> m_core;
    std::shared_ptr<ImmutableGpuBufferCache> m_cache;
};

namespace {

class MetalConstCacheRegistry {
public:
    static MetalConstCacheRegistry& instance() {
        static MetalConstCacheRegistry registry;
        return registry;
    }

    std::shared_ptr<MetalConstCacheContext> acquire(MetalDeviceHandle device) {
        const uintptr_t key = reinterpret_cast<uintptr_t>(device);
        std::lock_guard<std::mutex> lock(m_mutex);
        if (auto it = m_entries.find(key); it != m_entries.end()) {
            if (auto cached = it->second.lock()) {
                return cached;
            }
            m_entries.erase(it);
        }

        auto created = std::make_shared<MetalConstCacheContext>(device);
        m_entries[key] = created;
        return created;
    }

private:
    std::mutex m_mutex;
    std::unordered_map<uintptr_t, std::weak_ptr<MetalConstCacheContext>> m_entries;
};

}  // namespace

MetalConstCache::MetalConstCache(MetalAllocator& persistent_alloc, MetalCommandQueueHandle /*queue*/)
    : m_context(MetalConstCacheRegistry::instance().acquire(persistent_alloc.core().device())) {}

MetalConstCache::~MetalConstCache() = default;

MetalBuffer MetalConstCache::get_or_create(const std::string& key,
                                           const void* data,
                                           size_t bytes,
                                           const BufferDesc& desc) {
    OPENVINO_ASSERT(data && bytes > 0, "MetalConstCache: empty constant data");
    OPENVINO_ASSERT(m_context, "MetalConstCache: missing shared context");
    return m_context->cache()->get_or_create(key, data, bytes, desc.type, [&]() {
        return m_context->create_constant_buffer(data, bytes, desc);
    });
}

size_t MetalConstCache::total_bytes() const {
    return m_context ? m_context->cache()->total_bytes() : 0u;
}

const void* MetalConstCache::shared_cache_identity() const {
    return m_context.get();
}

}  // namespace gfx_plugin
}  // namespace ov
