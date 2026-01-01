// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <vector>

#include "openvino/runtime/profiling_info.hpp"
#include "openvino/gfx_plugin/profiling.hpp"
#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

class GfxProfiler {
public:
    virtual ~GfxProfiler() = default;

    virtual void set_config(const GfxProfilerConfig& cfg) = 0;
    virtual const GfxProfilerConfig& config() const = 0;

    virtual void begin_infer(size_t expected_samples) = 0;
    virtual void end_infer(GpuCommandBufferHandle command_buffer) = 0;

    virtual std::vector<ov::ProfilingInfo> export_ov() const = 0;
    virtual std::string export_extended_json() const { return {}; }

    virtual void* native_handle() = 0;
};

}  // namespace gfx_plugin
}  // namespace ov
