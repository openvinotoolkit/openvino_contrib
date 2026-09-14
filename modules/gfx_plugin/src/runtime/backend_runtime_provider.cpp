// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/backend_runtime_provider.hpp"

#include <mutex>
#include <optional>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

std::mutex& runtime_provider_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<BackendRuntimeProvider>& runtime_providers() {
    static std::vector<BackendRuntimeProvider> providers;
    return providers;
}

std::optional<BackendRuntimeProvider> find_runtime_provider(GpuBackend backend) {
    std::lock_guard<std::mutex> lock(runtime_provider_mutex());
    for (const auto& provider : runtime_providers()) {
        if (provider.backend == backend) {
            return provider;
        }
    }
    return std::nullopt;
}

}  // namespace

BackendRuntimeProviderRegistration::BackendRuntimeProviderRegistration(BackendRuntimeProvider provider) {
    register_backend_runtime_provider(provider);
}

void register_backend_runtime_provider(BackendRuntimeProvider provider) {
    std::lock_guard<std::mutex> lock(runtime_provider_mutex());
    auto& providers = runtime_providers();
    for (auto& registered : providers) {
        if (registered.backend == provider.backend) {
            registered = provider;
            return;
        }
    }
    providers.push_back(provider);
}

std::unique_ptr<BackendState> create_backend_state(const compiler::BackendTarget& target,
                                                   const ov::AnyMap& properties,
                                                   const ov::SoPtr<ov::IRemoteContext>& context) {
    const auto backend = target.backend();
    auto provider = find_runtime_provider(backend);
    if (!provider || !provider->create_state) {
        OPENVINO_THROW("GFX ", backend_to_string(backend), " backend has no runtime provider.");
    }
    return provider->create_state(target, properties, context);
}

void execute_backend_infer(const compiler::BackendTarget& target,
                           InferRequest& request,
                           const std::shared_ptr<const CompiledModel>& compiled_model) {
    const auto backend = target.backend();
    auto provider = find_runtime_provider(backend);
    if (!provider || !provider->execute_infer) {
        OPENVINO_THROW("GFX ", backend_to_string(backend), " backend has no infer executor.");
    }
    provider->execute_infer(request, compiled_model);
}

void register_backend_profiling_trace_sinks(const compiler::BackendTarget& target) {
    const auto backend = target.backend();
    auto provider = find_runtime_provider(backend);
    if (!provider) {
        OPENVINO_THROW("GFX ", backend_to_string(backend), " backend has no runtime provider.");
    }
    if (provider->register_profiling_trace_sinks) {
        provider->register_profiling_trace_sinks(target);
    }
}

}  // namespace gfx_plugin
}  // namespace ov
