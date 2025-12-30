// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

struct BackendState {
    virtual ~BackendState() = default;
    virtual GpuBackend backend() const = 0;
};

}  // namespace gfx_plugin
}  // namespace ov
