// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/profiling/signpost_trace_sink.hpp"

#include "runtime/gfx_profiling_report.hpp"
#include "runtime/gfx_profiling_trace_sink.hpp"

#import <os/signpost.h>

#include <mutex>
#include <string>

namespace ov {
namespace gfx_plugin {
namespace {

os_log_t gfx_signpost_log() {
    static os_log_t log = os_log_create("org.openvino.gfx", "profiling");
    return log;
}

void emit_signpost_segment(std::string_view backend,
                           const GfxProfilingSegmentEntry& segment) {
    os_log_t log = gfx_signpost_log();
    if (!os_signpost_enabled(log)) {
        return;
    }
    const std::string backend_name{backend};
    os_signpost_event_emit(
        log,
        OS_SIGNPOST_ID_EXCLUSIVE,
        "gfx.segment",
        "backend=%{public}s phase=%{public}s name=%{public}s cpu_us=%llu gpu_us=%llu inflight=%lld queue=%llu cmd=%llu",
        backend_name.c_str(),
        segment.phase.c_str(),
        segment.name.c_str(),
        static_cast<unsigned long long>(segment.cpu_us),
        static_cast<unsigned long long>(segment.gpu_us),
        static_cast<long long>(segment.inflight_slot),
        static_cast<unsigned long long>(segment.queue_id),
        static_cast<unsigned long long>(segment.cmd_buffer_id));
}

}  // namespace

void register_metal_signpost_trace_sink() {
    static std::once_flag once;
    std::call_once(once, [] {
        register_gfx_profiling_trace_sink("signpost", emit_signpost_segment);
        register_gfx_profiling_trace_sink("os_signpost", emit_signpost_segment);
    });
}

}  // namespace gfx_plugin
}  // namespace ov
