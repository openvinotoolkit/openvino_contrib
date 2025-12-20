// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

namespace ov {
namespace metal_plugin {

struct MetalMemoryStats {
    uint64_t bytes_allocated_total = 0;
    uint64_t bytes_in_freelist = 0;
    uint64_t bytes_live_transient = 0;
    uint64_t bytes_live_handles = 0;
    uint64_t bytes_persistent = 0;
    uint64_t peak_live = 0;

    uint64_t num_alloc_calls = 0;
    uint64_t num_reuse_hits = 0;

    uint64_t h2d_bytes = 0;
    uint64_t d2h_bytes = 0;
    uint64_t h2d_ops = 0;
    uint64_t d2h_ops = 0;
};

}  // namespace metal_plugin
}  // namespace ov
