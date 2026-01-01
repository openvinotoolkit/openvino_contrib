// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/shape.hpp"
#include "runtime/gpu_backend_base.hpp"

namespace ov {
namespace gfx_plugin {

inline KernelDispatch make_default_dispatch(const ov::Shape& shape, size_t tg0 = 1) {
    KernelDispatch dispatch{};
    dispatch.grid[0] = shape.size() > 0 ? shape[0] : 1;
    dispatch.grid[1] = shape.size() > 1 ? shape[1] : 1;
    dispatch.grid[2] = shape.size() > 2 ? shape[2] : 1;
    dispatch.threads_per_group[0] = tg0;
    dispatch.threads_per_group[1] = 1;
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
