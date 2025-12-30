// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#include "backends/metal/runtime/memory/buffer.hpp"

namespace ov {
namespace gfx_plugin {

struct MetalDeviceCaps {
    bool has_unified_memory = false;
    bool supports_heaps = true;
    bool supports_counter_sampling = false;
    size_t max_buffer_length = 0;

    bool prefer_private_intermediates = true;
    bool prefer_shared_io = true;
};

MetalDeviceCaps query_metal_device_caps(MetalDeviceHandle device);

}  // namespace gfx_plugin
}  // namespace ov
