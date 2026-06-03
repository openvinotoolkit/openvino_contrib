// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/plugin/compiled_model_backend.hpp"

#include "backends/opencl/plugin/compiled_model_state.hpp"
#include "backends/opencl/runtime/memory_api.hpp"
#include "openvino/core/except.hpp"
#include "runtime/backend_runtime_provider.hpp"
#include "runtime/gfx_remote_context.hpp"

namespace ov {
namespace gfx_plugin {

void execute_opencl_infer_request(InferRequest& request,
                                  const std::shared_ptr<const CompiledModel>& compiled_model);

namespace {

const BackendRuntimeProviderRegistration kOpenClRuntimeProviderRegistration({
    GpuBackend::OpenCL,
    create_opencl_backend_state,
    execute_opencl_infer_request,
    nullptr,
});

}  // namespace

std::unique_ptr<BackendState> create_opencl_backend_state(
    const ov::AnyMap&,
    const ov::SoPtr<ov::IRemoteContext>& context) {
    if (context) {
        auto gfx_context = std::dynamic_pointer_cast<GfxRemoteContext>(context._ptr);
        OPENVINO_ASSERT(gfx_context, "GFX OpenCL: remote context type mismatch");
        OPENVINO_ASSERT(gfx_context->backend() == GpuBackend::OpenCL,
                        "GFX OpenCL: remote context backend mismatch");
    }
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
