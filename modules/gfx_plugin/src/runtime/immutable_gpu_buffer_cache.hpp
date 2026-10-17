// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

uint64_t hash_immutable_gpu_buffer_bytes(const void* data, size_t bytes);
std::string make_immutable_gpu_buffer_cache_key(const std::string& logical_key,
                                                const void* data,
                                                size_t bytes,
                                                const ov::element::Type& type);

class ImmutableGpuBufferCache {
public:
    using ReleaseFn = std::function<void(GpuBuffer&&)>;

    explicit ImmutableGpuBufferCache(ReleaseFn release = {});
    ~ImmutableGpuBufferCache();

    GpuBuffer get_or_create(const std::string& logical_key,
                            const void* data,
                            size_t bytes,
                            ov::element::Type type,
                            const std::function<GpuBuffer()>& factory);

    size_t entry_count() const;
    size_t total_bytes() const;

private:
    struct Entry {
        GpuBuffer buffer{};
        size_t bytes = 0;
        ov::element::Type type = ov::element::dynamic;
    };

    void release_buffer(GpuBuffer&& buffer) const;

    ReleaseFn m_release;
    mutable std::mutex m_mutex;
    std::unordered_map<std::string, Entry> m_entries;
    size_t m_total_bytes = 0;
};

}  // namespace gfx_plugin
}  // namespace ov
