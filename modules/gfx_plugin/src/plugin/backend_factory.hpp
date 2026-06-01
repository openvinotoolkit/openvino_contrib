// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "plugin/backend_state.hpp"
#include "runtime/gfx_backend_utils.hpp"

namespace ov {
namespace gfx_plugin {

struct BackendRuntimeProvider {
    GpuBackend backend = GpuBackend::Unknown;
    std::unique_ptr<BackendState> (*create_state)(const ov::AnyMap& properties,
                                                  const ov::SoPtr<ov::IRemoteContext>& context) = nullptr;
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
void register_backend_profiling_trace_sinks(GpuBackend backend);

}  // namespace gfx_plugin
}  // namespace ov
