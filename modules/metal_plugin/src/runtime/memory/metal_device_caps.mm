// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/memory/metal_device_caps.hpp"

namespace ov {
namespace metal_plugin {

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
    return caps;
}

}  // namespace metal_plugin
}  // namespace ov
