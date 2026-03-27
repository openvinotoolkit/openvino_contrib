// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>

#include "openvino/core/type/element_type.hpp"
#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

struct GpuExecutionDeviceInfo {
    GpuBackend backend = GpuBackend::Metal;
    std::string device_key;
    uint32_t preferred_simd_width = 1;
    uint32_t subgroup_size = 1;
    uint32_t max_total_threads_per_group = 1;
    std::array<uint32_t, 3> max_threads_per_group{{1, 1, 1}};
};

// Backend-neutral buffer manager interface (backend implementations derive from this).
class GpuBufferManager {
public:
    virtual ~GpuBufferManager() = default;

    virtual std::optional<GpuExecutionDeviceInfo> query_execution_device_info() const {
        return std::nullopt;
    }

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
