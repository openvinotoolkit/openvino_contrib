// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "runtime/memory/metal_buffer.hpp"

namespace ov {
namespace gfx_plugin {

struct FreeKey {
    uint32_t bucket = 0;
    MetalStorage storage = MetalStorage::Private;
    uint32_t options_mask = 0;

    bool operator<(const FreeKey& other) const {
        if (bucket != other.bucket)
            return bucket < other.bucket;
        if (storage != other.storage)
            return static_cast<int>(storage) < static_cast<int>(other.storage);
        return options_mask < other.options_mask;
    }
};

class MetalFreeList {
public:
    MetalBuffer try_pop(const FreeKey& key) {
        auto it = m_free.find(key);
        if (it == m_free.end() || it->second.empty())
            return {};
        MetalBuffer buf = std::move(it->second.back());
        it->second.pop_back();
        if (buf.size > 0 && m_bytes_cached >= buf.size) {
            m_bytes_cached -= buf.size;
        }
        return buf;
    }

    void push(const FreeKey& key, MetalBuffer&& buf) {
        m_bytes_cached += buf.size;
        m_free[key].push_back(std::move(buf));
    }

    void trim(size_t max_cached_bytes);

    uint64_t bytes_in_freelist() const { return m_bytes_cached; }

private:
    std::map<FreeKey, std::vector<MetalBuffer>> m_free;
    uint64_t m_bytes_cached = 0;
};

inline void MetalFreeList::trim(size_t max_cached_bytes) {
    if (m_bytes_cached <= max_cached_bytes)
        return;
    for (auto it = m_free.begin(); it != m_free.end(); ++it) {
        auto& vec = it->second;
        while (!vec.empty() && m_bytes_cached > max_cached_bytes) {
            auto& buf = vec.back();
            if (buf.size <= m_bytes_cached)
                m_bytes_cached -= buf.size;
            vec.pop_back();
        }
        if (m_bytes_cached <= max_cached_bytes) {
            break;
        }
    }
}

}  // namespace gfx_plugin
}  // namespace ov
