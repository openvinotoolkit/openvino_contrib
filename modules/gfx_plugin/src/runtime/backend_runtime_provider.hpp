// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "runtime/backend_runtime.hpp"
#include "runtime/gfx_backend_utils.hpp"

namespace ov {
namespace gfx_plugin {

class CompiledModel;
class InferRequest;

struct BackendRuntimeProvider {
    GpuBackend backend = GpuBackend::Unknown;
    std::unique_ptr<BackendState> (*create_state)(const ov::AnyMap& properties,
                                                  const ov::SoPtr<ov::IRemoteContext>& context) = nullptr;
    void (*execute_infer)(InferRequest& request,
                          const std::shared_ptr<const CompiledModel>& compiled_model) = nullptr;
    void (*register_profiling_trace_sinks)() = nullptr;
};

class BackendRuntimeProviderRegistration final {
public:
    explicit BackendRuntimeProviderRegistration(BackendRuntimeProvider provider);
};

void register_backend_runtime_provider(BackendRuntimeProvider provider);

std::unique_ptr<BackendState> create_backend_state(GpuBackend backend,
                                                   const ov::AnyMap& properties,
                                                   const ov::SoPtr<ov::IRemoteContext>& context);
void execute_backend_infer(GpuBackend backend,
                           InferRequest& request,
                           const std::shared_ptr<const CompiledModel>& compiled_model);
void register_backend_profiling_trace_sinks(GpuBackend backend);

}  // namespace gfx_plugin
}  // namespace ov
