// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "backends/metal/runtime/memory/allocator_core.hpp"
#include "backends/metal/runtime/memory/buffer.hpp"

namespace ov {
namespace gfx_plugin {

class MetalHeapPool {
public:
    explicit MetalHeapPool(MetalAllocatorCore& core);
    ~MetalHeapPool();

    MetalBuffer alloc_private_from_heap(const BufferDesc& desc);
    void on_release(MetalBuffer& buf);

private:
    MetalAllocatorCore& m_core;
    std::vector<MetalHeapHandle> m_private_heaps;
};

}  // namespace gfx_plugin
}  // namespace ov
