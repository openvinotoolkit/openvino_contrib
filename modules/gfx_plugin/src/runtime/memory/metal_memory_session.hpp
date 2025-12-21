// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "runtime/memory/metal_allocator.hpp"

namespace ov {
namespace gfx_plugin {

struct ScratchKey {
    size_t id = 0;
    bool operator==(const ScratchKey& other) const { return id == other.id; }
};

struct ScratchKeyHash {
    size_t operator()(const ScratchKey& k) const { return std::hash<size_t>{}(k.id); }
};

class MetalMemorySession {
public:
    MetalMemorySession(MetalAllocator& alloc, void* profiler = nullptr, void* exec_ctx = nullptr);

    MetalBuffer alloc_transient(const BufferDesc& desc);
    MetalBuffer ensure_scratch(BufferHandle& handle, const BufferDesc& desc, bool persistent = false);

    void end();
    const MetalMemoryStats& stats() const { return m_alloc.stats(); }
    MetalAllocator& allocator() { return m_alloc; }

private:
    MetalAllocator& m_alloc;
    std::vector<MetalBuffer> m_transient;
    void* m_profiler = nullptr;
    void* m_exec_ctx = nullptr;
    bool m_active = true;
};

}  // namespace gfx_plugin
}  // namespace ov
