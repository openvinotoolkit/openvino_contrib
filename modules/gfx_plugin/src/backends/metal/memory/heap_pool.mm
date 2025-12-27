// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/memory/heap_pool.hpp"

#include <algorithm>

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
constexpr size_t kAlignment = 256;
constexpr size_t kSmallHeap = 32 * 1024 * 1024;   // 32MB
constexpr size_t kLargeHeap = 256 * 1024 * 1024;  // 256MB

size_t bucket_size(size_t size) {
    const size_t aligned = size == 0 ? kAlignment : ((size + kAlignment - 1) & ~(kAlignment - 1));
    size_t b = kAlignment;
    while (b < aligned) {
        b <<= 1;
    }
    return b;
}

}  // namespace

MetalHeapPool::MetalHeapPool(MetalAllocatorCore& core) : m_core(core) {}

MetalHeapPool::~MetalHeapPool() {
#ifdef __OBJC__
    for (auto heap : m_private_heaps) {
        if (heap) {
            [static_cast<id<MTLHeap>>(heap) release];
        }
    }
#endif
    m_private_heaps.clear();
}

MetalBuffer MetalHeapPool::alloc_private_from_heap(const BufferDesc& desc) {
    MetalBuffer out;
#ifdef __OBJC__
    if (desc.storage != MetalStorage::Private) {
        return out;
    }
    const size_t bucket = bucket_size(desc.bytes);
    for (auto heap : m_private_heaps) {
        if (!heap) continue;
        id<MTLHeap> mheap = static_cast<id<MTLHeap>>(heap);
        if ([mheap maxAvailableSizeWithAlignment:kAlignment] < bucket)
            continue;
        out = m_core.create_buffer_from_heap(heap, desc);
        if (out.valid()) {
            out.heap = heap;
            return out;
        }
    }

    const size_t heap_size = bucket <= kSmallHeap ? kSmallHeap : std::max(kLargeHeap, bucket * 2);
    MetalHeapHandle heap = m_core.create_heap(MetalStorage::Private, heap_size, /*options_mask=*/0);
    if (!heap) {
        return out;
    }
    m_private_heaps.push_back(heap);
    out = m_core.create_buffer_from_heap(heap, desc);
    if (out.valid()) {
        out.heap = heap;
    }
#else
    (void)desc;
#endif
    return out;
}

void MetalHeapPool::on_release(MetalBuffer& buf) {
#ifdef __OBJC__
    if (!buf.heap) {
        return;
    }
    auto mb = static_cast<id<MTLBuffer>>(buf.buffer);
    if (mb && [mb respondsToSelector:@selector(makeAliasable)]) {
        [mb makeAliasable];
    }
#endif
}

}  // namespace gfx_plugin
}  // namespace ov
