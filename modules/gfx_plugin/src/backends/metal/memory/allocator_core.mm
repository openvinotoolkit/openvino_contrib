// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/memory/allocator_core.hpp"

#include <algorithm>

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
constexpr size_t kAlignment = 256;

#ifdef __OBJC__
MTLResourceOptions options_from_desc(const BufferDesc& desc) {
    MTLResourceOptions opts = 0;
    const auto storage = desc.storage == MetalStorage::Private ? MTLResourceStorageModePrivate
                                                               : MTLResourceStorageModeShared;
    opts |= storage;

    if (desc.write_combined) {
        opts |= MTLResourceCPUCacheModeWriteCombined;
    } else {
        opts |= MTLResourceCPUCacheModeDefaultCache;
    }
    return opts;
}
#endif

}  // namespace

MetalAllocatorCore::MetalAllocatorCore(MetalDeviceHandle device, MetalDeviceCaps caps)
    : m_device(device), m_caps(caps) {
#ifdef __OBJC__
    if (!m_device) {
        OPENVINO_THROW("MetalAllocatorCore: device is null");
    }
#endif
}

MetalBuffer MetalAllocatorCore::create_buffer(const BufferDesc& desc) {
    MetalBuffer out;
#ifdef __OBJC__
    auto dev = static_cast<id<MTLDevice>>(m_device);
    OPENVINO_ASSERT(dev, "MetalAllocatorCore: device is null");
    const size_t aligned = desc.bytes == 0 ? kAlignment : ((desc.bytes + kAlignment - 1) & ~(kAlignment - 1));
    MTLResourceOptions opts = options_from_desc(desc);
    id<MTLBuffer> buf = [dev newBufferWithLength:aligned options:opts];
    OPENVINO_ASSERT(buf, "MetalAllocatorCore: newBufferWithLength failed");
    out.buffer = buf;
    out.size = aligned;
    out.type = desc.type;
    out.storage_mode = static_cast<uint32_t>(buf.storageMode);
    out.options_mask = static_cast<uint32_t>(opts);
    out.persistent = false;
    out.backend = GpuBackend::Metal;
    out.host_visible = (out.storage_mode == static_cast<uint32_t>(MTLStorageModeShared));
#else
    (void)desc;
    OPENVINO_THROW("MetalAllocatorCore::create_buffer requires Objective-C++ (Metal)");
#endif
    return out;
}

MetalBuffer MetalAllocatorCore::create_buffer_from_heap(MetalHeapHandle heap, const BufferDesc& desc) {
    MetalBuffer out;
#ifdef __OBJC__
    auto mheap = static_cast<id<MTLHeap>>(heap);
    OPENVINO_ASSERT(mheap, "MetalAllocatorCore: heap is null");
    const size_t aligned = desc.bytes == 0 ? kAlignment : ((desc.bytes + kAlignment - 1) & ~(kAlignment - 1));
    MTLResourceOptions opts = options_from_desc(desc);
    id<MTLBuffer> buf = [mheap newBufferWithLength:aligned options:opts];
    if (!buf) {
        return out;
    }
    out.buffer = buf;
    out.size = aligned;
    out.type = desc.type;
    out.heap = heap;
    out.storage_mode = static_cast<uint32_t>(buf.storageMode);
    out.options_mask = static_cast<uint32_t>(opts);
    out.persistent = false;
    out.backend = GpuBackend::Metal;
    out.host_visible = (out.storage_mode == static_cast<uint32_t>(MTLStorageModeShared));
#else
    (void)heap;
    (void)desc;
    OPENVINO_THROW("MetalAllocatorCore::create_buffer_from_heap requires Objective-C++ (Metal)");
#endif
    return out;
}

MetalHeapHandle MetalAllocatorCore::create_heap(MetalStorage storage, size_t heap_bytes, uint32_t options_mask) {
#ifdef __OBJC__
    auto dev = static_cast<id<MTLDevice>>(m_device);
    OPENVINO_ASSERT(dev, "MetalAllocatorCore: device is null");
    MTLHeapDescriptor* desc = [MTLHeapDescriptor new];
    desc.storageMode = storage == MetalStorage::Private ? MTLStorageModePrivate : MTLStorageModeShared;
    desc.cpuCacheMode = (options_mask & MTLResourceCPUCacheModeWriteCombined) ? MTLCPUCacheModeWriteCombined
                                                                              : MTLCPUCacheModeDefaultCache;
    desc.hazardTrackingMode = MTLHazardTrackingModeTracked;
    desc.size = heap_bytes;
    id<MTLHeap> heap = [dev newHeapWithDescriptor:desc];
    [desc release];
    return heap;
#else
    (void)storage;
    (void)heap_bytes;
    (void)options_mask;
    OPENVINO_THROW("MetalAllocatorCore::create_heap requires Objective-C++ (Metal)");
#endif
}

MetalBuffer MetalAllocatorCore::wrap_shared(void* ptr, size_t bytes, ov::element::Type type) {
    MetalBuffer out;
#ifdef __OBJC__
    if (!ptr || bytes == 0) {
        return out;
    }
    auto dev = static_cast<id<MTLDevice>>(m_device);
    OPENVINO_ASSERT(dev, "MetalAllocatorCore: device is null");
    id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:ptr
                                               length:bytes
                                              options:MTLResourceStorageModeShared
                                          deallocator:^(void*, NSUInteger) {
                                          }];
    OPENVINO_ASSERT(buf, "MetalAllocatorCore: failed to wrap shared memory");
    out.buffer = buf;
    out.size = bytes;
    out.type = type;
    out.storage_mode = static_cast<uint32_t>(MTLStorageModeShared);
    out.options_mask = static_cast<uint32_t>(MTLResourceStorageModeShared);
    out.external = true;
    out.from_handle = true;
    out.backend = GpuBackend::Metal;
    out.host_visible = true;
#else
    (void)ptr;
    (void)bytes;
    (void)type;
    OPENVINO_THROW("MetalAllocatorCore::wrap_shared requires Objective-C++ (Metal)");
#endif
    return out;
}

void MetalAllocatorCore::release_buffer(MetalBuffer& buf) {
#ifdef __OBJC__
    if (!buf.buffer)
        return;
    auto mb = static_cast<id<MTLBuffer>>(buf.buffer);
    [mb release];
    buf.buffer = nullptr;
#else
    (void)buf;
#endif
}

}  // namespace gfx_plugin
}  // namespace ov
