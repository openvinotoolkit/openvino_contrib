// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#include "backends/metal/memory/buffer.hpp"
#include "backends/metal/memory/device_caps.hpp"

namespace ov {
namespace gfx_plugin {

class MetalAllocatorCore {
public:
    explicit MetalAllocatorCore(MetalDeviceHandle device, MetalDeviceCaps caps);

    MetalBuffer create_buffer(const BufferDesc& desc);
    MetalBuffer create_buffer_from_heap(MetalHeapHandle heap, const BufferDesc& desc);
    MetalHeapHandle create_heap(MetalStorage storage, size_t heap_bytes, uint32_t options_mask);

    MetalBuffer wrap_shared(void* ptr, size_t bytes, ov::element::Type type);
    MetalBuffer wrap_shared(const void* ptr, size_t bytes, ov::element::Type type) {
        return wrap_shared(const_cast<void*>(ptr), bytes, type);
    }
    void release_buffer(MetalBuffer& buf);

    MetalDeviceHandle device() const { return m_device; }
    const MetalDeviceCaps& caps() const { return m_caps; }

private:
    MetalDeviceHandle m_device = nullptr;
    MetalDeviceCaps m_caps;
};

}  // namespace gfx_plugin
}  // namespace ov
