// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>

namespace ov {
namespace gfx_plugin {

struct ParallelDispatchConfig {
    bool enabled = false;
    size_t loop_dims = 0;
    uint32_t tile_h = 1;
    uint32_t tile_w = 1;
    uint32_t threads_h = 1;
    uint32_t threads_w = 1;
    uint32_t channel_block = 1;
};

struct KernelDispatch {
    size_t grid[3] = {1, 1, 1};
    size_t threads_per_group[3] = {1, 1, 1};
};

}  // namespace gfx_plugin
}  // namespace ov
