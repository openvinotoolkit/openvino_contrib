// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/gfx_kernel_cache.hpp"

namespace ov {
namespace gfx_plugin {

GfxKernelCache& GfxKernelCache::instance() {
    static GfxKernelCache cache;
    return cache;
}

std::shared_ptr<ICompiledKernel> GfxKernelCache::lookup(const KernelCacheKey& key) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_cache.find(key);
    if (it == m_cache.end()) {
        return {};
    }
    auto existing = it->second.lock();
    if (!existing) {
        m_cache.erase(it);
        return {};
    }
    return existing;
}

void GfxKernelCache::store(const KernelCacheKey& key, const std::shared_ptr<ICompiledKernel>& kernel) {
    if (!kernel) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    m_cache[key] = kernel;
}

}  // namespace gfx_plugin
}  // namespace ov
