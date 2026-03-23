// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstdint>

#include "openvino/core/shape.hpp"
#include "runtime/gpu_backend_base.hpp"

namespace ov {
namespace gfx_plugin {

inline KernelDispatch make_default_dispatch(const ov::Shape& shape, size_t tg0 = 1) {
    KernelDispatch dispatch{};
    uint64_t total = 1;
    for (auto d : shape) {
        total *= static_cast<uint64_t>(d);
    }
    dispatch.grid[0] = std::max<uint64_t>(total, 1);
    dispatch.grid[1] = 1;
    dispatch.grid[2] = 1;
    dispatch.threads_per_group[0] = tg0;
    dispatch.threads_per_group[1] = 1;
    dispatch.threads_per_group[2] = 1;
    return dispatch;
}

struct ParallelDispatchConfig {
    bool enabled = false;
    size_t loop_dims = 0;
    uint32_t tile_h = 1;
    uint32_t tile_w = 1;
    uint32_t threads_h = 1;
    uint32_t threads_w = 1;
};

inline KernelDispatch make_parallel_dispatch(const ov::Shape& shape,
                                             const ParallelDispatchConfig& cfg,
                                             const ICompiledKernel* kernel = nullptr) {
    KernelDispatch dispatch{};
    if (!cfg.enabled) {
        return dispatch;
    }
    const size_t rank = shape.size();
    const uint64_t tile_h = cfg.tile_h ? cfg.tile_h : 1;
    const uint64_t tile_w = cfg.tile_w ? cfg.tile_w : 1;
    const uint64_t thread_h = cfg.threads_h ? cfg.threads_h : 1;
    const uint64_t thread_w = cfg.threads_w ? cfg.threads_w : 1;
    const bool wants_threads = (thread_h > 1 || thread_w > 1);
    if (rank == 1) {
        dispatch.grid[0] = wants_threads ? (shape[0] * thread_w) : shape[0];
    } else if (rank == 2) {
        dispatch.grid[0] = wants_threads ? (shape[0] * thread_w) : shape[0];
        dispatch.grid[1] = wants_threads ? (shape[1] * thread_h) : shape[1];
    } else if (rank >= 3) {
        const uint64_t c = shape[rank - 3];
        const uint64_t h = shape[rank - 2];
        const uint64_t w = shape[rank - 1];
        const uint64_t h_tiles = (h + tile_h - 1) / tile_h;
        const uint64_t w_tiles = (w + tile_w - 1) / tile_w;
        if (wants_threads && cfg.loop_dims >= 3) {
            const size_t block_dims = cfg.loop_dims > 2 ? cfg.loop_dims - 2 : 0;
            if (cfg.loop_dims == 3) {
                dispatch.grid[0] = c * thread_w;
                dispatch.grid[1] = h_tiles * thread_h;
                dispatch.grid[2] = w_tiles;
            } else if (block_dims >= 3) {
                dispatch.grid[0] = c * thread_w;
                dispatch.grid[1] = h_tiles * thread_h;
                dispatch.grid[2] = w_tiles;
            } else if (block_dims == 2) {
                dispatch.grid[0] = h_tiles * thread_h;
                dispatch.grid[1] = w_tiles * thread_w;
                dispatch.grid[2] = 1;
            } else {
                dispatch.grid[0] = (c > 1) ? c : (h_tiles * w_tiles);
                dispatch.grid[1] = 1;
                dispatch.grid[2] = 1;
            }
        } else {
            dispatch.grid[0] = c * thread_w;
            dispatch.grid[1] = h_tiles * thread_h;
            dispatch.grid[2] = w_tiles;
        }
    }
    dispatch.threads_per_group[0] =
        kernel ? kernel->clamp_threadgroup_size(thread_w) : std::max<uint64_t>(thread_w, 1);
    dispatch.threads_per_group[1] = thread_h;
    dispatch.threads_per_group[2] = 1;
    return dispatch;
}

inline KernelDispatch make_1d_dispatch(size_t x, size_t tg0) {
    KernelDispatch dispatch{};
    dispatch.grid[0] = x;
    dispatch.threads_per_group[0] = tg0;
    return dispatch;
}

inline KernelDispatch make_2d_dispatch(size_t x, size_t y, size_t tg0, size_t tg1 = 1) {
    KernelDispatch dispatch{};
    dispatch.grid[0] = x;
    dispatch.grid[1] = y;
    dispatch.threads_per_group[0] = tg0;
    dispatch.threads_per_group[1] = tg1;
    return dispatch;
}

inline KernelDispatch make_3d_dispatch(size_t x,
                                       size_t y,
                                       size_t z,
                                       size_t tg0,
                                       size_t tg1 = 1,
                                       size_t tg2 = 1) {
    KernelDispatch dispatch{};
    dispatch.grid[0] = x;
    dispatch.grid[1] = y;
    dispatch.grid[2] = z;
    dispatch.threads_per_group[0] = tg0;
    dispatch.threads_per_group[1] = tg1;
    dispatch.threads_per_group[2] = tg2;
    return dispatch;
}

}  // namespace gfx_plugin
}  // namespace ov
