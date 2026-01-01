// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <string_view>
#include <utility>

#include "openvino/gfx_plugin/compiled_model.hpp"
#include "openvino/core/except.hpp"
#include "plugin/backend_state.hpp"
#include "plugin/gfx_profiling_utils.hpp"
#include "plugin/infer_request_state.hpp"
#include "runtime/gfx_profiler.hpp"

namespace ov {
namespace gfx_plugin {

template <typename PreEndFn>
inline void finalize_infer_profiling(std::string_view backend,
                                     const std::shared_ptr<const CompiledModel>& cm,
                                     InferRequestState& state,
                                     GfxProfiler* profiler,
                                     GpuCommandBufferHandle command_buffer,
                                     PreEndFn&& pre_end) {
    state.last_profiling.clear();
    if (profiler) {
        std::forward<PreEndFn>(pre_end)();
        profiler->end_infer(command_buffer);
        state.last_profiling = profiler->export_ov();
    }
    if (cm) {
        const auto level = profiler ? cm->profiling_level() : ProfilingLevel::Off;
        const auto extended = profiler ? profiler->export_extended_json() : std::string{};
        cm->update_last_profiling_report_json(
            build_profiling_report_json(backend, level, state.last_profiling, extended));
    }
}

inline void finalize_infer_profiling(std::string_view backend,
                                     const std::shared_ptr<const CompiledModel>& cm,
                                     InferRequestState& state,
                                     GfxProfiler* profiler,
                                     GpuCommandBufferHandle command_buffer) {
    finalize_infer_profiling(backend, cm, state, profiler, command_buffer, []() {});
}

inline GfxProfiler* prepare_infer_profiler(const CompiledModel& cm,
                                           InferRequestState& state,
                                           const char* error_prefix) {
    if (!cm.enable_profiling()) {
        state.profiler_cfg = {};
        return nullptr;
    }
    state.profiler_cfg = make_profiler_config(cm.profiling_level());
    if (state.profiler_cfg.level == ProfilingLevel::Off) {
        return nullptr;
    }
    OPENVINO_ASSERT(state.backend, error_prefix, ": backend infer state is null");
    auto* backend_state = cm.backend_state();
    OPENVINO_ASSERT(backend_state, error_prefix, ": backend state is null");
    if (!state.backend->profiler) {
        state.backend->profiler = backend_state->create_profiler(state.profiler_cfg);
    }
    if (!state.backend->profiler) {
        return nullptr;
    }
    state.backend->profiler->set_config(state.profiler_cfg);
    return state.backend->profiler.get();
}

}  // namespace gfx_plugin
}  // namespace ov
