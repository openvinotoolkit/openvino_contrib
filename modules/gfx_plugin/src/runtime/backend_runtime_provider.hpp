// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "compiler/backend_target.hpp"
#include "runtime/backend_runtime.hpp"
#include "common/gfx_backend_utils.hpp"

namespace ov {
namespace gfx_plugin {

class CompiledModel;
class InferRequest;

struct BackendRuntimeProvider {
    GpuBackend backend = GpuBackend::Unknown;
    std::unique_ptr<BackendState> (*create_state)(const compiler::BackendTarget& target,
                                                  const ov::AnyMap& properties,
                                                  const ov::SoPtr<ov::IRemoteContext>& context) = nullptr;
    void (*execute_infer)(InferRequest& request,
                          const std::shared_ptr<const CompiledModel>& compiled_model) = nullptr;
    void (*register_profiling_trace_sinks)(const compiler::BackendTarget& target) = nullptr;
};

class BackendRuntimeProviderRegistration final {
public:
    explicit BackendRuntimeProviderRegistration(BackendRuntimeProvider provider);
};

void register_backend_runtime_provider(BackendRuntimeProvider provider);

std::unique_ptr<BackendState> create_backend_state(const compiler::BackendTarget& target,
                                                   const ov::AnyMap& properties,
                                                   const ov::SoPtr<ov::IRemoteContext>& context);
void execute_backend_infer(const compiler::BackendTarget& target,
                           InferRequest& request,
                           const std::shared_ptr<const CompiledModel>& compiled_model);
void register_backend_profiling_trace_sinks(const compiler::BackendTarget& target);

}  // namespace gfx_plugin
}  // namespace ov
