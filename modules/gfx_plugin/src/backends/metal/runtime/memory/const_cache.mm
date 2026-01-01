// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/memory/const_cache.hpp"

#include "backends/metal/runtime/metal_memory.hpp"
#include "openvino/core/except.hpp"

#include <cstring>

namespace ov {
namespace gfx_plugin {

MetalConstCache::MetalConstCache(MetalAllocator& persistent_alloc, MetalCommandQueueHandle queue)
    : m_alloc(persistent_alloc), m_queue(queue) {}

MetalConstCache::~MetalConstCache() {
    for (auto& kv : m_cache) {
        m_alloc.core().release_buffer(kv.second);
    }
    m_cache.clear();
}

const MetalBuffer& MetalConstCache::get_or_create(const ConstKey& key,
                                                  const void* data,
                                                  size_t bytes,
                                                  const BufferDesc& desc) {
    auto it = m_cache.find(key);
    if (it != m_cache.end()) {
        return it->second;
    }
    OPENVINO_ASSERT(data && bytes > 0, "MetalConstCache: empty constant data");

    MetalBuffer buf;
    if (desc.storage == MetalStorage::Shared) {
        buf = m_alloc.core().wrap_shared(const_cast<void*>(data), bytes, desc.type);
        buf.persistent = true;
        buf.external = true;
        buf.from_handle = true;
    } else {
        OPENVINO_ASSERT(m_queue, "GFX: const cache requires command queue for private storage");
        BufferDesc local = desc;
        local.storage = MetalStorage::Private;
        buf = m_alloc.allocate(local, true);

        BufferDesc staging_desc;
        staging_desc.bytes = bytes;
        staging_desc.type = desc.type;
        staging_desc.storage = MetalStorage::Shared;
        staging_desc.usage = BufferUsage::Staging;
        staging_desc.write_combined = true;

        MetalBuffer staging = m_alloc.core().create_buffer(staging_desc);
        OPENVINO_ASSERT(staging.valid(), "GFX: failed to allocate staging buffer for constants");
        if (bytes) {
            void* mapped = metal_map_buffer(staging);
            OPENVINO_ASSERT(mapped, "GFX: failed to map staging buffer for constants");
            std::memcpy(mapped, data, bytes);
            metal_unmap_buffer(staging);
            metal_copy_buffer(m_queue, staging, buf, bytes);
        }
        m_alloc.core().release_buffer(staging);
    }
    m_total_bytes += bytes;
    auto inserted = m_cache.emplace(key, std::move(buf));
    return inserted.first->second;
}

}  // namespace gfx_plugin
}  // namespace ov
