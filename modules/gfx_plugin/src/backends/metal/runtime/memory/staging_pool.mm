// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/memory/staging_pool.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
constexpr size_t kAlignment = 256;

size_t bucket_size(size_t size) {
    const size_t aligned = size == 0 ? kAlignment : ((size + kAlignment - 1) & ~(kAlignment - 1));
    size_t b = kAlignment;
    while (b < aligned) {
        b <<= 1;
    }
    return b;
}

}  // namespace

MetalStagingPool::MetalStagingPool(MetalAllocatorCore& core) : m_core(core) {}

MetalBuffer MetalStagingPool::allocate(size_t bytes, const char* label) {
    const size_t bucket = bucket_size(bytes);
    BufferDesc desc;
    desc.bytes = bucket;
    desc.storage = MetalStorage::Shared;
    desc.usage = BufferUsage::Staging;
    desc.write_combined = true;
    desc.label = label;

    uint32_t options_mask = 0;
#ifdef __OBJC__
    MTLResourceOptions opts = MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
    options_mask = static_cast<uint32_t>(opts);
#endif
    FreeKey key{static_cast<uint32_t>(bucket), MetalStorage::Shared, options_mask};
    MetalBuffer buf = m_free.try_pop(key);
    if (buf.valid()) {
        return buf;
    }
    buf = m_core.create_buffer(desc);
    buf.options_mask = options_mask;
    return buf;
}

void MetalStagingPool::release(MetalBuffer&& buf) {
    if (!buf.valid())
        return;
    const size_t bucket = bucket_size(buf.size);
    FreeKey key{static_cast<uint32_t>(bucket), MetalStorage::Shared, buf.options_mask};
    m_free.push(key, std::move(buf));
}

}  // namespace gfx_plugin
}  // namespace ov
