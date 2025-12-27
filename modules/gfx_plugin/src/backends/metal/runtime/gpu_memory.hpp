// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "runtime/gpu_memory.hpp"
#include "backends/metal/runtime/memory.hpp"
#include "backends/metal/memory/allocator.hpp"
#include "backends/metal/memory/device_caps.hpp"

namespace ov {
namespace gfx_plugin {

class MetalGpuAllocator final : public IGpuAllocator {
public:
    MetalGpuAllocator(MetalAllocator& alloc, MetalAllocatorCore& core, const MetalDeviceCaps& caps);

    GpuBackend backend() const override { return GpuBackend::Metal; }
    GpuBuffer allocate(const GpuBufferDesc& desc) override;
    GpuBuffer wrap_shared(void* ptr, size_t bytes, ov::element::Type type) override;
    void release(GpuBuffer&& buf) override;

private:
    MetalAllocator& m_alloc;
    MetalAllocatorCore& m_core;
    MetalDeviceCaps m_caps{};
};

}  // namespace gfx_plugin
}  // namespace ov
