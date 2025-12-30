// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "backends/metal/runtime/memory/allocator.hpp"

namespace ov {
namespace gfx_plugin {

struct ConstKey {
    std::string id;
    bool operator==(const ConstKey& other) const { return id == other.id; }
};

struct ConstKeyHash {
    size_t operator()(const ConstKey& k) const { return std::hash<std::string>{}(k.id); }
};

class MetalConstCache {
public:
    explicit MetalConstCache(MetalAllocator& persistent_alloc);
    ~MetalConstCache();

    const MetalBuffer& get_or_create(const ConstKey& key, const void* data, size_t bytes, const BufferDesc& desc);
    size_t total_bytes() const { return m_total_bytes; }

private:
    MetalAllocator& m_alloc;
    std::unordered_map<ConstKey, MetalBuffer, ConstKeyHash> m_cache;
    size_t m_total_bytes = 0;
};

}  // namespace gfx_plugin
}  // namespace ov
