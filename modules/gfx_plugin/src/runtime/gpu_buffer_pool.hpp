// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <utility>

#include "runtime/gpu_memory.hpp"

namespace ov {
namespace gfx_plugin {

// Lightweight handle-based reuse for GPU buffers across infer iterations.
class GpuBufferPool {
public:
    explicit GpuBufferPool(IGpuAllocator& allocator) : m_allocator(allocator) {}

    GpuBuffer ensure(BufferHandle& handle, const GpuBufferDesc& desc) {
        if (desc.bytes == 0) {
            release(handle);
            return {};
        }

        const bool want_host_visible = desc.cpu_read || desc.cpu_write || !desc.prefer_device_local;
        if (handle.valid() &&
            handle.capacity >= desc.bytes &&
            handle.buf.type == desc.type &&
            handle.buf.host_visible == want_host_visible) {
            handle.buf.backend = m_allocator.backend();
            handle.buf.from_handle = true;
            return handle.buf;
        }

        release(handle);
        GpuBuffer buf = m_allocator.allocate(desc);
        buf.from_handle = true;
        handle.buf = buf;
        handle.capacity = buf.size;
        return handle.buf;
    }

    void release(BufferHandle& handle) {
        if (handle.valid()) {
            m_allocator.release(std::move(handle.buf));
        }
        handle.capacity = 0;
    }

private:
    IGpuAllocator& m_allocator;
};

}  // namespace gfx_plugin
}  // namespace ov
