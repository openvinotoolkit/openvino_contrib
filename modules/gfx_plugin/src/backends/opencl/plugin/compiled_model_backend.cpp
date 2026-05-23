// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/plugin/compiled_model_backend.hpp"

#include "backends/opencl/plugin/compiled_model_state.hpp"
#include "backends/opencl/runtime/memory_api.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<BackendState> create_opencl_backend_state(
    const ov::AnyMap&,
    const ov::SoPtr<ov::IRemoteContext>& context) {
    OPENVINO_ASSERT(!context, "GFX OpenCL: remote context import is not implemented yet");
    ensure_opencl_memory_ops_registered();
    ensure_opencl_stage_factory_registered();
    auto runtime = OpenClRuntimeContext::instance();
    auto state = std::make_unique<OpenClBackendState>();
    state->context = runtime;
    state->const_manager = std::make_shared<OpenClBufferManager>(runtime);
    return state;
}

std::unique_ptr<GfxProfiler> OpenClBackendState::create_profiler(const GfxProfilerConfig&) const {
    return {};
}

}  // namespace gfx_plugin
}  // namespace ov
