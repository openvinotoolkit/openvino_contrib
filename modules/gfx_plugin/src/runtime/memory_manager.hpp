// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "runtime/gpu_types.hpp"

namespace ov {
namespace gfx_plugin {

struct GpuBufferDesc {
    size_t bytes = 0;
    ov::element::Type type = ov::element::dynamic;
    BufferUsage usage = BufferUsage::Intermediate;

    bool cpu_read = false;
    bool cpu_write = false;
    bool prefer_device_local = true;
    const char* label = nullptr;
};

inline void validate_gpu_buffer_desc(const GpuBufferDesc& desc, const char* error_prefix = "GFX") {
    if (desc.bytes == 0) {
        return;
    }
    switch (desc.usage) {
    case BufferUsage::Staging:
        OPENVINO_ASSERT(!desc.prefer_device_local,
                        error_prefix,
                        ": staging buffers must be host-visible");
        OPENVINO_ASSERT(desc.cpu_read || desc.cpu_write,
                        error_prefix,
                        ": staging buffers require CPU access");
        break;
    case BufferUsage::IO:
        break;
    case BufferUsage::Const:
        OPENVINO_ASSERT(!desc.cpu_read,
                        error_prefix,
                        ": const buffers are not expected to be CPU-readable");
        OPENVINO_ASSERT(desc.prefer_device_local || desc.cpu_write,
                        error_prefix,
                        ": const buffers must be device-local or host-uploadable");
        break;
    case BufferUsage::Intermediate:
    case BufferUsage::Temp:
        OPENVINO_ASSERT(desc.prefer_device_local,
                        error_prefix,
                        ": internal buffers must be device-local");
        OPENVINO_ASSERT(!desc.cpu_read && !desc.cpu_write,
                        error_prefix,
                        ": internal buffers must be device-only");
        break;
    default:
        break;
    }
}

class IGpuAllocator {
public:
    virtual ~IGpuAllocator() = default;
    virtual GpuBackend backend() const = 0;
    virtual GpuBuffer allocate(const GpuBufferDesc& desc) = 0;
    virtual GpuBuffer wrap_shared(void* ptr, size_t bytes, ov::element::Type type) = 0;
    virtual void release(GpuBuffer&& buf) = 0;
};

// Unified map/unmap and host copy helpers.
void* gpu_map_buffer(const GpuBuffer& buf);
void gpu_unmap_buffer(const GpuBuffer& buf);
void gpu_copy_from_host(GpuBuffer& buf, const void* src, size_t bytes, size_t offset = 0);
void gpu_copy_to_host(const GpuBuffer& buf, void* dst, size_t bytes, size_t offset = 0);
void gpu_copy_buffer(GpuCommandQueueHandle queue,
                     const GpuBuffer& src,
                     const GpuBuffer& dst,
                     size_t bytes);
bool gpu_host_visible(const GpuBuffer& buf);

}  // namespace gfx_plugin
}  // namespace ov
