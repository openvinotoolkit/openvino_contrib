// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <string_view>

#include "runtime/gfx_profiling_report.hpp"

namespace ov {
namespace gfx_plugin {

namespace detail {
inline GfxProfilingTrace*& compile_trace_slot() {
    static thread_local GfxProfilingTrace* slot = nullptr;
    return slot;
}

inline std::string& compile_scope_name_slot() {
    static thread_local std::string slot;
    return slot;
}
}  // namespace detail

class ScopedCompileProfilingContext {
public:
    ScopedCompileProfilingContext(GfxProfilingTrace* trace, std::string_view scope_name)
        : m_prev_trace(detail::compile_trace_slot()),
          m_prev_scope(detail::compile_scope_name_slot()) {
        detail::compile_trace_slot() = trace;
        detail::compile_scope_name_slot().assign(scope_name.data(), scope_name.size());
    }

    ~ScopedCompileProfilingContext() {
        detail::compile_trace_slot() = m_prev_trace;
        detail::compile_scope_name_slot() = std::move(m_prev_scope);
    }

private:
    GfxProfilingTrace* m_prev_trace = nullptr;
    std::string m_prev_scope;
};

inline GfxProfilingTrace* current_compile_trace() {
    return detail::compile_trace_slot();
}

inline std::string make_compile_event_name(std::string_view local_name) {
    const auto& scope_name = detail::compile_scope_name_slot();
    if (scope_name.empty()) {
        return std::string(local_name);
    }
    std::string full = scope_name;
    full += "::";
    full.append(local_name.data(), local_name.size());
    return full;
}

inline void add_compile_segment(std::string_view local_name, uint64_t cpu_us) {
    if (auto* trace = current_compile_trace()) {
        trace->add_segment("compile", make_compile_event_name(local_name), cpu_us);
    }
}

inline void increment_compile_counter(std::string_view name, uint64_t delta = 1) {
    if (auto* trace = current_compile_trace()) {
        trace->increment_counter(name, delta);
    }
}

inline void set_compile_counter(std::string_view name, uint64_t value) {
    if (auto* trace = current_compile_trace()) {
        trace->set_counter(name, value);
    }
}

}  // namespace gfx_plugin
}  // namespace ov
