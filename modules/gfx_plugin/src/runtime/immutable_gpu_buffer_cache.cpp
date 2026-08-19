// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/immutable_gpu_buffer_cache.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

uint64_t hash_immutable_gpu_buffer_bytes(const void* data, size_t bytes) {
    constexpr uint64_t kOffset = 1469598103934665603ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffset;
    const auto* ptr = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= static_cast<uint64_t>(ptr[i]);
        hash *= kPrime;
    }
    return hash;
}

std::string make_immutable_gpu_buffer_cache_key(const std::string& logical_key,
                                                const void* data,
                                                size_t bytes,
                                                const ov::element::Type& type) {
    std::ostringstream os;
    os << logical_key
       << "#bytes=" << bytes
       << "#type=" << type.get_type_name()
       << "#hash=" << hash_immutable_gpu_buffer_bytes(data, bytes);
    return os.str();
}

ImmutableGpuBufferCache::ImmutableGpuBufferCache(ReleaseFn release) : m_release(std::move(release)) {}

ImmutableGpuBufferCache::~ImmutableGpuBufferCache() {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& kv : m_entries) {
        if (kv.second.buffer.valid()) {
            release_buffer(std::move(kv.second.buffer));
        }
    }
    m_entries.clear();
    m_total_bytes = 0;
}

GpuBuffer ImmutableGpuBufferCache::get_or_create(const std::string& logical_key,
                                                 const void* data,
                                                 size_t bytes,
                                                 ov::element::Type type,
                                                 const std::function<GpuBuffer()>& factory) {
    if (bytes == 0) {
        return {};
    }
    OPENVINO_ASSERT(data, "GFX: immutable buffer cache requires non-null data");
    OPENVINO_ASSERT(factory, "GFX: immutable buffer cache requires factory");

    const std::string storage_key = make_immutable_gpu_buffer_cache_key(logical_key, data, bytes, type);
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (auto it = m_entries.find(storage_key); it != m_entries.end()) {
            return it->second.buffer;
        }
    }

    GpuBuffer created = factory();
    OPENVINO_ASSERT(created.valid(), "GFX: immutable buffer cache factory returned invalid buffer");

    std::lock_guard<std::mutex> lock(m_mutex);
    if (auto it = m_entries.find(storage_key); it != m_entries.end()) {
        release_buffer(std::move(created));
        return it->second.buffer;
    }

    auto& slot = m_entries[storage_key];
    slot.buffer = created;
    slot.bytes = bytes;
    slot.type = type;
    m_total_bytes += bytes;
    return slot.buffer;
}

size_t ImmutableGpuBufferCache::entry_count() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_entries.size();
}

size_t ImmutableGpuBufferCache::total_bytes() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_total_bytes;
}

void ImmutableGpuBufferCache::release_buffer(GpuBuffer&& buffer) const {
    if (buffer.valid() && m_release) {
        m_release(std::move(buffer));
    }
}

}  // namespace gfx_plugin
}  // namespace ov
