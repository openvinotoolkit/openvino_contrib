// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/memory/metal_memory_session.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

MetalMemorySession::MetalMemorySession(MetalAllocator& alloc, void* profiler, void* exec_ctx)
    : m_alloc(alloc), m_profiler(profiler), m_exec_ctx(exec_ctx) {}

MetalBuffer MetalMemorySession::alloc_transient(const BufferDesc& desc) {
    OPENVINO_ASSERT(m_active, "MetalMemorySession: alloc_transient on inactive session");
    MetalBuffer buf = m_alloc.allocate(desc, /*persistent=*/false);
    m_transient.push_back(buf);
    return buf;
}

MetalBuffer MetalMemorySession::ensure_scratch(BufferHandle& handle, const BufferDesc& desc, bool persistent) {
    OPENVINO_ASSERT(m_active, "MetalMemorySession: ensure_scratch on inactive session");
    return m_alloc.ensure_handle(handle, desc, persistent);
}

void MetalMemorySession::end() {
    if (!m_active)
        return;
    for (auto& buf : m_transient) {
        m_alloc.release(std::move(buf));
    }
    m_transient.clear();
    m_active = false;
}

}  // namespace metal_plugin
}  // namespace ov
