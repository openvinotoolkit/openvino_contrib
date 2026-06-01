// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/backend_factory.hpp"

#include <mutex>
#include <optional>
#include <vector>

#include "compiler/backend_registry.hpp"
#include "openvino/core/except.hpp"
#include "runtime/gfx_backend_utils.hpp"

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

std::unique_ptr<BackendState> create_backend_state(GpuBackend backend,
                                                   const ov::AnyMap& properties,
                                                   const ov::SoPtr<ov::IRemoteContext>& context) {
    if (!compiler::BackendRegistry::default_registry().resolve(backend)) {
        OPENVINO_THROW("GFX ", backend_to_string(backend), " backend is not available in this build.");
    }
    auto provider = find_runtime_provider(backend);
    if (!provider || !provider->create_state) {
        OPENVINO_THROW("GFX ", backend_to_string(backend),
                       " backend has a compiler module but no runtime provider.");
    }
    return provider->create_state(properties, context);
}

void register_backend_profiling_trace_sinks(GpuBackend backend) {
    if (!compiler::BackendRegistry::default_registry().resolve(backend)) {
        return;
    }
    auto provider = find_runtime_provider(backend);
    if (!provider) {
        OPENVINO_THROW("GFX ", backend_to_string(backend),
                       " backend has a compiler module but no runtime provider.");
    }
    if (provider->register_profiling_trace_sinks) {
        provider->register_profiling_trace_sinks();
    }
}

}  // namespace gfx_plugin
}  // namespace ov
