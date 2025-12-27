// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>

#include "backends/metal/memory/allocator_core.hpp"
#include "backends/metal/memory/freelist.hpp"

namespace ov {
namespace gfx_plugin {

class MetalStagingPool {
public:
    explicit MetalStagingPool(MetalAllocatorCore& core);

    MetalBuffer allocate(size_t bytes, const char* label = nullptr);
    void release(MetalBuffer&& buf);

private:
    MetalAllocatorCore& m_core;
    MetalFreeList m_free;
};

}  // namespace gfx_plugin
}  // namespace ov
