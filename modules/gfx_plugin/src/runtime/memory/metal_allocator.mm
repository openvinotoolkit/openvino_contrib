// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/memory/metal_allocator.hpp"

#include <algorithm>
#include <string>

#include "openvino/core/except.hpp"
#include "runtime/profiling/metal_profiler.hpp"

#include <chrono>

namespace ov {
namespace gfx_plugin {

namespace {
constexpr size_t kAlignment = 256;

const char* usage_name(BufferUsage usage) {
    switch (usage) {
    case BufferUsage::IO:
        return "io";
    case BufferUsage::Const:
        return "const";
    case BufferUsage::Intermediate:
        return "intermediate";
    case BufferUsage::Temp:
        return "temp";
    case BufferUsage::Staging:
        return "staging";
    default:
        return "unknown";
    }
}

const char* storage_name(MetalStorage storage) {
    switch (storage) {
    case MetalStorage::Private:
        return "private";
    case MetalStorage::Shared:
        return "shared";
    default:
        return "unknown";
    }
}

uint32_t options_mask_from_desc(const BufferDesc& desc) {
#ifdef __OBJC__
    MTLResourceOptions opts = 0;
    opts |= (desc.storage == MetalStorage::Private) ? MTLResourceStorageModePrivate
                                                    : MTLResourceStorageModeShared;
    opts |= desc.write_combined ? MTLResourceCPUCacheModeWriteCombined : MTLResourceCPUCacheModeDefaultCache;
    return static_cast<uint32_t>(opts);
#else
    (void)desc;
    return 0;
#endif
}

}  // namespace

MetalAllocator::MetalAllocator(MetalAllocatorCore& core,
                               MetalHeapPool& heaps,
                               MetalFreeList& freelist,
                               MetalStagingPool& staging,
                               MetalDeviceCaps caps)
    : m_core(core), m_heaps(heaps), m_freelist(freelist), m_staging(staging), m_caps(caps) {}

void MetalAllocator::set_profiler(MetalProfiler* profiler, bool detailed) {
    m_profiler = profiler;
    m_profile_allocations = (profiler != nullptr) && detailed;
}

size_t MetalAllocator::align_size(size_t size) const {
    const size_t mask = kAlignment - 1;
    if (size == 0)
        return kAlignment;
    return (size + mask) & ~mask;
}

size_t MetalAllocator::bucket_size(size_t size) const {
    size = align_size(size);
    size_t b = kAlignment;
    while (b < size) {
        b <<= 1;
    }
    return b;
}

FreeKey MetalAllocator::make_key(size_t bucket, const BufferDesc& desc, uint32_t options_mask) const {
    FreeKey key;
    key.bucket = static_cast<uint32_t>(bucket);
    key.storage = desc.storage;
    key.options_mask = options_mask;
    return key;
}

MetalBuffer MetalAllocator::allocate(const BufferDesc& desc, bool persistent) {
    auto start = m_profile_allocations ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
    BufferDesc local = desc;
    local.bytes = bucket_size(desc.bytes);
    uint32_t options_mask = options_mask_from_desc(desc);
    FreeKey key = make_key(local.bytes, desc, options_mask);

    if (!persistent) {
        MetalBuffer cached = m_freelist.try_pop(key);
        if (cached.valid()) {
            cached.persistent = false;
            cached.from_handle = false;
            cached.type = desc.type;
            m_stats.num_reuse_hits += 1;
            m_stats.bytes_in_freelist = m_freelist.bytes_in_freelist();
            if (m_profile_allocations && m_profiler) {
                const auto end = std::chrono::steady_clock::now();
                std::string tag;
                const char* label = desc.label;
                if (!label) {
                    tag = std::string(usage_name(desc.usage)) + "/" + storage_name(desc.storage);
                    label = tag.c_str();
                }
                m_profiler->record_alloc(label,
                                         cached.size,
                                         true,
                                         std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            }
            return cached;
        }
    }

    MetalBuffer buf;
    if (desc.storage == MetalStorage::Private && m_caps.supports_heaps && !persistent) {
        buf = m_heaps.alloc_private_from_heap(local);
        if (!buf.valid()) {
            buf = m_core.create_buffer(local);
        }
    } else {
        buf = m_core.create_buffer(local);
    }

    buf.persistent = persistent;
    buf.type = desc.type;
    buf.options_mask = options_mask;

    m_stats.num_alloc_calls += 1;
    m_stats.bytes_allocated_total += buf.size;
    if (persistent) {
        m_stats.bytes_persistent += buf.size;
    }
    if (m_profile_allocations && m_profiler) {
        const auto end = std::chrono::steady_clock::now();
        std::string tag;
        const char* label = desc.label;
        if (!label) {
            tag = std::string(usage_name(desc.usage)) + "/" + storage_name(desc.storage);
            label = tag.c_str();
        }
        m_profiler->record_alloc(label,
                                 buf.size,
                                 false,
                                 std::chrono::duration_cast<std::chrono::microseconds>(end - start));
    }
    return buf;
}

MetalBuffer MetalAllocator::allocate_staging(size_t bytes, const char* label) {
    return m_staging.allocate(bytes, label);
}

MetalBuffer MetalAllocator::ensure_handle(BufferHandle& handle, const BufferDesc& desc, bool persistent) {
    auto start = m_profile_allocations ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
    const size_t target = bucket_size(desc.bytes);
    const bool storage_matches = !handle.buf.valid() ||
#ifdef __OBJC__
                                 (desc.storage == MetalStorage::Private
                                      ? handle.buf.storage_mode == static_cast<uint32_t>(MTLStorageModePrivate)
                                      : handle.buf.storage_mode == static_cast<uint32_t>(MTLStorageModeShared));
#else
                                 true;
#endif
    if (handle.capacity >= target && handle.buf.valid() && storage_matches) {
        handle.buf.from_handle = true;
        handle.buf.type = desc.type;
        m_stats.num_reuse_hits += 1;
        if (m_profile_allocations && m_profiler) {
            const auto end = std::chrono::steady_clock::now();
            std::string tag;
            const char* label = desc.label;
            if (!label) {
                tag = std::string(usage_name(desc.usage)) + "/" + storage_name(desc.storage);
                label = tag.c_str();
            }
            m_profiler->record_alloc(label,
                                     handle.buf.size,
                                     true,
                                     std::chrono::duration_cast<std::chrono::microseconds>(end - start));
        }
        return handle.buf;
    }

    if (handle.buf.valid()) {
        MetalBuffer old = handle.buf;
        old.from_handle = false;
        release(std::move(old));
    }

    size_t grow = handle.capacity == 0 ? target : std::max(target, static_cast<size_t>(handle.capacity * 3 / 2));
    BufferDesc tmp = desc;
    tmp.bytes = grow;
    handle.buf = allocate(tmp, persistent);
    handle.buf.from_handle = true;
    handle.capacity = handle.buf.size;
    if (m_profile_allocations && m_profiler) {
        const auto end = std::chrono::steady_clock::now();
        std::string tag;
        const char* label = desc.label;
        if (!label) {
            tag = std::string(usage_name(desc.usage)) + "/" + storage_name(desc.storage);
            label = tag.c_str();
        }
        m_profiler->record_alloc(label,
                                 handle.buf.size,
                                 false,
                                 std::chrono::duration_cast<std::chrono::microseconds>(end - start));
    }
    return handle.buf;
}

void MetalAllocator::release(MetalBuffer&& buf) {
    if (!buf.valid() || buf.persistent || buf.external) {
        return;
    }
    if (buf.from_handle) {
        return;
    }
    if (buf.heap) {
        m_heaps.on_release(buf);
    }
    const size_t bucket = bucket_size(buf.size);
    BufferDesc desc;
    desc.bytes = buf.size;
#ifdef __OBJC__
    if (buf.storage_mode == static_cast<uint32_t>(MTLStorageModePrivate)) {
        desc.storage = MetalStorage::Private;
    } else {
        desc.storage = MetalStorage::Shared;
    }
#else
    desc.storage = MetalStorage::Shared;
#endif
    FreeKey key = make_key(bucket, desc, buf.options_mask);
    m_freelist.push(key, std::move(buf));
    m_stats.bytes_in_freelist = m_freelist.bytes_in_freelist();
}

}  // namespace gfx_plugin
}  // namespace ov
