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

class GfxProfiler;

enum class GpuDeviceFamily {
    Generic,
    Apple,
    QualcommAdreno,
    BroadcomV3D,
};

inline const char* gpu_device_family_name(GpuDeviceFamily family) {
    switch (family) {
    case GpuDeviceFamily::Apple:
        return "apple";
    case GpuDeviceFamily::QualcommAdreno:
        return "adreno";
    case GpuDeviceFamily::BroadcomV3D:
        return "broadcom_v3d";
    case GpuDeviceFamily::Generic:
    default:
        return "generic";
    }
}

struct GpuExecutionDeviceInfo {
    GpuBackend backend = GpuBackend::Metal;
    GpuDeviceFamily device_family = GpuDeviceFamily::Generic;
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
    virtual void begin_const_upload_batch() {}
    virtual void flush_const_upload_batch(GpuCommandBufferHandle /*command_buffer*/,
                                          GfxProfiler* /*profiler*/) {}
    virtual void end_const_upload_batch() {}
};

}  // namespace gfx_plugin
}  // namespace ov
