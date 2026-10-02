// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "runtime/gpu_buffer.hpp"
#include "runtime/gpu_memory_ops.hpp"

namespace ov {
namespace gfx_plugin {

size_t opencl_allocation_bytes(size_t bytes, ov::element::Type type);
void* opencl_map_buffer(const GpuBuffer& buf);
void opencl_unmap_buffer(const GpuBuffer& buf);
void opencl_flush_buffer(const GpuBuffer& buf, size_t bytes, size_t offset = 0);
void opencl_invalidate_buffer(const GpuBuffer& buf, size_t bytes, size_t offset = 0);
void opencl_free_buffer(GpuBuffer& buf);
void opencl_copy_buffer(GpuCommandQueueHandle execution_context,
                        const GpuBuffer& src,
                        const GpuBuffer& dst,
                        size_t bytes);
void opencl_copy_buffer_regions(GpuCommandQueueHandle execution_context,
                                const GpuBuffer& src,
                                const GpuBuffer& dst,
                                const GpuBufferCopyRegion* regions,
                                size_t region_count);
void ensure_opencl_memory_ops_registered();

}  // namespace gfx_plugin
}  // namespace ov
