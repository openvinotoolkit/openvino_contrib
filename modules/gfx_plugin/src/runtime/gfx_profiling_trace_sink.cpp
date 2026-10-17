// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_profiling_trace_sink.hpp"

#include <mutex>
#include <string>
#include <unordered_map>

namespace ov {
namespace gfx_plugin {
namespace {

std::unordered_map<std::string, GfxProfilingTraceSinkEmitter>& trace_sinks() {
    static auto* sinks =
        new std::unordered_map<std::string, GfxProfilingTraceSinkEmitter>();
    return *sinks;
}

std::mutex& trace_sinks_mutex() {
    static auto* mutex = new std::mutex();
    return *mutex;
}

}  // namespace

void register_gfx_profiling_trace_sink(std::string_view name,
                                       GfxProfilingTraceSinkEmitter emitter) {
    if (name.empty() || !emitter) {
        return;
    }
    std::lock_guard<std::mutex> lock(trace_sinks_mutex());
    trace_sinks()[std::string{name}] = emitter;
}

bool gfx_profiling_trace_sink_available(std::string_view name) {
    if (name.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(trace_sinks_mutex());
    return trace_sinks().find(std::string{name}) != trace_sinks().end();
}

void emit_gfx_profiling_trace_sink(std::string_view name,
                                   std::string_view backend,
                                   const GfxProfilingSegmentEntry& segment) {
    GfxProfilingTraceSinkEmitter emitter = nullptr;
    {
        std::lock_guard<std::mutex> lock(trace_sinks_mutex());
        const auto it = trace_sinks().find(std::string{name});
        if (it != trace_sinks().end()) {
            emitter = it->second;
        }
    }
    if (emitter) {
        emitter(backend, segment);
    }
}

}  // namespace gfx_plugin
}  // namespace ov
