// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

namespace ov {
namespace gfx_plugin {

// Backend-neutral buffer manager interface (backend implementations derive from this).
class GpuBufferManager {
public:
    virtual ~GpuBufferManager() = default;
};

}  // namespace gfx_plugin
}  // namespace ov
