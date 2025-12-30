// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

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
