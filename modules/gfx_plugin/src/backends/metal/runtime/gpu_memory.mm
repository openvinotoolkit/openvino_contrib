// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/gpu_memory.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

MetalGpuAllocator::MetalGpuAllocator(MetalAllocator& alloc, MetalAllocatorCore& core, const MetalDeviceCaps& caps)
    : m_alloc(alloc), m_core(core), m_caps(caps) {}

GpuBuffer MetalGpuAllocator::allocate(const GpuBufferDesc& desc) {
    BufferDesc mdesc;
    mdesc.bytes = desc.bytes;
    mdesc.type = desc.type;
    mdesc.usage = desc.usage;
    mdesc.label = desc.label;
    mdesc.cpu_read = desc.cpu_read;
    mdesc.cpu_write = desc.cpu_write;

    const bool want_shared = desc.cpu_read || desc.cpu_write || !desc.prefer_device_local;
    if (want_shared || !m_caps.prefer_private_intermediates) {
        mdesc.storage = MetalStorage::Shared;
    } else {
        mdesc.storage = MetalStorage::Private;
    }

    GpuBuffer buf = m_alloc.allocate(mdesc, /*persistent=*/false);
    if (!buf.buffer && desc.bytes > 0) {
        OPENVINO_THROW("GFX Metal: failed to allocate buffer");
    }
    buf.backend = GpuBackend::Metal;
    buf.host_visible = (mdesc.storage == MetalStorage::Shared);
    return buf;
}

GpuBuffer MetalGpuAllocator::wrap_shared(void* ptr, size_t bytes, ov::element::Type type) {
    GpuBuffer buf = m_core.wrap_shared(ptr, bytes, type);
    buf.backend = GpuBackend::Metal;
    buf.host_visible = true;
    return buf;
}

void MetalGpuAllocator::release(GpuBuffer&& buf) {
    m_alloc.release(std::move(buf));
}

}  // namespace gfx_plugin
}  // namespace ov
