// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/plugin/compiled_model_backend.hpp"

#include "backends/opencl/plugin/compiled_model_state.hpp"
#include "backends/opencl/runtime/opencl_api.hpp"
#include "backends/opencl/runtime/memory_api.hpp"
#include "openvino/core/except.hpp"
#include "plugin/gfx_property_utils.hpp"
#include "runtime/backend_runtime_provider.hpp"
#include "runtime/gfx_remote_context.hpp"

namespace ov {
namespace gfx_plugin {

void execute_opencl_infer_request(InferRequest& request,
                                  const std::shared_ptr<const CompiledModel>& compiled_model);

namespace {

compiler::BackendTarget resolve_opencl_backend_target(
    const ov::AnyMap&,
    const BackendRequest&) {
    const auto& api = OpenClApi::instance();
    const auto selection = select_opencl_gpu_device(api);
    const auto info = make_opencl_execution_device_info(selection);
    return compiler::BackendTarget::from_backend_device_family(
        GpuBackend::OpenCL, info.device_family);
}

const BackendTargetResolverRegistration kOpenClTargetResolverRegistration({
    GpuBackend::OpenCL,
    resolve_opencl_backend_target,
});

const BackendRuntimeProviderRegistration kOpenClRuntimeProviderRegistration({
    GpuBackend::OpenCL,
    create_opencl_backend_state,
    execute_opencl_infer_request,
    nullptr,
});

}  // namespace

std::unique_ptr<BackendState> create_opencl_backend_state(
    const compiler::BackendTarget& target,
    const ov::AnyMap&,
    const ov::SoPtr<ov::IRemoteContext>& context) {
    OPENVINO_ASSERT(target.backend() == GpuBackend::OpenCL,
                    "GFX OpenCL: backend state target mismatch: ",
                    target.debug_string());
    if (context) {
        auto gfx_context = std::dynamic_pointer_cast<GfxRemoteContext>(context._ptr);
        OPENVINO_ASSERT(gfx_context, "GFX OpenCL: remote context type mismatch");
        OPENVINO_ASSERT(gfx_context->target().is_compatible_with_fingerprint(target.fingerprint()),
                        "GFX OpenCL: remote context target mismatch");
    }
    ensure_opencl_memory_ops_registered();
    ensure_opencl_stage_factory_registered();
    auto runtime = OpenClRuntimeContext::instance();
    const auto runtime_info = runtime->execution_device_info();
    const auto runtime_target = compiler::BackendTarget::from_backend_device_family(
        GpuBackend::OpenCL,
        runtime_info.device_family);
    OPENVINO_ASSERT(runtime_target.is_compatible_with_fingerprint(target.fingerprint()),
                    "GFX OpenCL: runtime device target mismatch. Compiled target: ",
                    target.debug_string(),
                    "; runtime target: ",
                    runtime_target.debug_string());
    auto state = std::make_unique<OpenClBackendState>();
    state->runtime_target = target;
    state->context = runtime;
    state->const_manager = std::make_shared<OpenClBufferManager>(runtime);
    return state;
}

std::unique_ptr<GfxProfiler> OpenClBackendState::create_profiler(const GfxProfilerConfig&) const {
    return {};
}

}  // namespace gfx_plugin
}  // namespace ov
