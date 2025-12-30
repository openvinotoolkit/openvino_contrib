// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

namespace ov {
namespace gfx_plugin {

enum class ProfilingLevel : uint8_t {
    Off = 0,
    Standard = 1,
    Detailed = 2,
};

struct GfxProfilerConfig {
    ProfilingLevel level = ProfilingLevel::Off;
    bool include_transfers = true;
    bool include_allocations = true;
    bool include_segments = true;

    // When counters are unavailable, do not fall back to CPU-derived GPU times.
    // Per-op GPU timing is available only when counters are supported.
    bool force_cmd_buffer_per_op = false;
};

}  // namespace gfx_plugin
}  // namespace ov
