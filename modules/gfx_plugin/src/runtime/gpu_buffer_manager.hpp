// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <string>

#include "openvino/core/type/element_type.hpp"
#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

// Backend-neutral buffer manager interface (backend implementations derive from this).
class GpuBufferManager {
public:
    virtual ~GpuBufferManager() = default;

    virtual bool supports_const_cache() const { return false; }
    virtual GpuBuffer wrap_const(const std::string& /*key*/,
                                 const void* /*data*/,
                                 size_t /*bytes*/,
                                 ov::element::Type /*type*/) {
        return {};
    }
};

}  // namespace gfx_plugin
}  // namespace ov
