// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string_view>

namespace ov {
namespace gfx_plugin {

struct GfxProfilingSegmentEntry;

using GfxProfilingTraceSinkEmitter =
    void (*)(std::string_view backend, const GfxProfilingSegmentEntry& segment);

void register_gfx_profiling_trace_sink(std::string_view name,
                                       GfxProfilingTraceSinkEmitter emitter);
bool gfx_profiling_trace_sink_available(std::string_view name);
void emit_gfx_profiling_trace_sink(std::string_view name,
                                   std::string_view backend,
                                   const GfxProfilingSegmentEntry& segment);

}  // namespace gfx_plugin
}  // namespace ov
