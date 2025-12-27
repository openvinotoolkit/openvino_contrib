// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/memory/const_cache.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

MetalConstCache::MetalConstCache(MetalAllocator& persistent_alloc) : m_alloc(persistent_alloc) {}

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

    if (desc.storage != MetalStorage::Shared) {
        OPENVINO_THROW("GFX: const cache requires shared storage (no CPU copies)");
    }

    MetalBuffer buf = m_alloc.core().wrap_shared(const_cast<void*>(data), bytes, desc.type);
    buf.persistent = true;
    buf.external = true;
    buf.from_handle = true;
    m_total_bytes += bytes;
    auto inserted = m_cache.emplace(key, std::move(buf));
    return inserted.first->second;
}

}  // namespace gfx_plugin
}  // namespace ov
