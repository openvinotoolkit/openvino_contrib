// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include "backends/metal/runtime/memory/allocator.hpp"
#include "runtime/immutable_gpu_buffer_cache.hpp"

namespace ov {
namespace gfx_plugin {

class MetalConstCacheContext;

class MetalConstCache {
public:
    MetalConstCache(MetalAllocator& persistent_alloc, MetalCommandQueueHandle queue);
    ~MetalConstCache();

    MetalBuffer get_or_create(const std::string& key, const void* data, size_t bytes, const BufferDesc& desc);
    size_t total_bytes() const;
    const void* shared_cache_identity() const;

private:
    std::shared_ptr<MetalConstCacheContext> m_context;
};

}  // namespace gfx_plugin
}  // namespace ov
