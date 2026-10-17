// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/memory/device_caps.hpp"

namespace ov {
namespace gfx_plugin {

MetalDeviceCaps query_metal_device_caps(MetalDeviceHandle device) {
    MetalDeviceCaps caps{};
#ifdef __OBJC__
    auto dev = static_cast<id<MTLDevice>>(device);
    if (!dev) {
        return caps;
    }
    if ([dev respondsToSelector:@selector(hasUnifiedMemory)]) {
        caps.has_unified_memory = [dev hasUnifiedMemory];
    }
    if ([dev respondsToSelector:@selector(maxBufferLength)]) {
        caps.max_buffer_length = static_cast<size_t>(dev.maxBufferLength);
    }
    // Heaps are supported on macOS Metal devices; avoid deprecated feature queries.
    caps.supports_heaps = true;
    if ([dev respondsToSelector:@selector(supportsCounterSampling:)]) {
        caps.supports_counter_sampling = [dev supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
    }
#endif
    // Policy defaults.
    caps.prefer_private_intermediates = true;
    caps.prefer_shared_io = caps.has_unified_memory;
    // Metal does not expose thread execution width on MTLDevice directly.
    // Keep a device-derived conservative profile here and refine per-kernel later.
    caps.preferred_simd_width = 32;
    caps.max_total_threads_per_threadgroup = caps.has_unified_memory ? 1024u : 512u;
    caps.max_threads_per_threadgroup_x = caps.max_total_threads_per_threadgroup;
    caps.max_threads_per_threadgroup_y = caps.max_total_threads_per_threadgroup;
    caps.max_threads_per_threadgroup_z = 64u;
    return caps;
}

}  // namespace gfx_plugin
}  // namespace ov
