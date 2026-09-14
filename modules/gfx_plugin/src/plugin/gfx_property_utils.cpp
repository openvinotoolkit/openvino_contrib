// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gfx_property_utils.hpp"

#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

#include "compiler/backend_registry.hpp"
#include "openvino/core/except.hpp"
#include "plugin/gfx_profiling_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

std::vector<BackendTargetResolverProvider>& target_resolver_providers() {
    static std::vector<BackendTargetResolverProvider> providers;
    return providers;
}

std::mutex& target_resolver_mutex() {
    static std::mutex mutex;
    return mutex;
}

bool backend_registered(GpuBackend backend) {
    return backend_supported(backend);
}

bool find_target_resolver_provider(GpuBackend backend,
                                   BackendTargetResolverProvider& provider) {
    std::lock_guard<std::mutex> lock(target_resolver_mutex());
    for (const auto& candidate : target_resolver_providers()) {
        if (candidate.backend == backend && candidate.resolve) {
            provider = candidate;
            return true;
        }
    }
    return false;
}

bool exact_target_registered(const compiler::BackendTarget& target,
                             compiler::BackendTarget& registered_target) {
    const auto module =
        compiler::BackendRegistry::default_registry().resolve(target);
    if (!module) {
        return false;
    }
    registered_target = module->target();
    return registered_target.is_compatible_with_fingerprint(target.fingerprint());
}

compiler::BackendTarget resolve_backend_target_for_request(
    const ov::AnyMap& properties,
    const BackendRequest& request,
    GpuBackend backend) {
    compiler::BackendTarget target = compiler::BackendTarget::from_backend(backend);
    BackendTargetResolverProvider provider;
    if (find_target_resolver_provider(backend, provider)) {
        target = provider.resolve(properties, request);
    }
    if (target.backend() != backend) {
        OPENVINO_THROW("GFX: backend target resolver for '",
                       backend_to_string(backend),
                       "' returned incompatible target: ",
                       target.debug_string());
    }

    compiler::BackendTarget registered_target;
    if (!exact_target_registered(target, registered_target)) {
        OPENVINO_THROW("GFX: backend target is not registered: ",
                       target.debug_string(),
                       ". Concrete BackendTarget modules must be registered explicitly.");
    }
    return registered_target;
}

}  // namespace

BackendTargetResolverRegistration::BackendTargetResolverRegistration(
    BackendTargetResolverProvider provider) {
    OPENVINO_ASSERT(backend_known(provider.backend),
                    "GFX: BackendTarget resolver provider requires a known backend");
    OPENVINO_ASSERT(provider.resolve,
                    "GFX: BackendTarget resolver provider callback is null");
    std::lock_guard<std::mutex> lock(target_resolver_mutex());
    auto& providers = target_resolver_providers();
    for (auto& candidate : providers) {
        if (candidate.backend == provider.backend) {
            candidate = provider;
            return;
        }
    }
    providers.push_back(std::move(provider));
}

BackendRequest get_backend_request(const ov::AnyMap& properties) {
    if (auto it = properties.find(kGfxBackendProperty); it != properties.end()) {
        const auto requested = ov::util::to_lower(it->second.as<std::string>());
        return BackendRequest{parse_backend_kind(requested), true, requested};
    }
    return BackendRequest{default_backend_kind(), false, ""};
}

GpuBackend resolve_backend_kind_from_properties(const ov::AnyMap& properties,
                                                bool log_fallback,
                                                const char* log_tag) {
    auto request = get_backend_request(properties);
    if (backend_registered(request.kind)) {
        return request.kind;
    }
    if (request.explicit_request) {
        OPENVINO_THROW("GFX: backend '", request.requested, "' is not available on this platform");
    }
    (void)log_fallback;
    (void)log_tag;
    const auto default_backend = default_backend_kind();
    OPENVINO_THROW("GFX: default backend '",
                   backend_to_string(default_backend),
                   "' is not available on this platform");
}

std::string resolve_backend_name_from_properties(const ov::AnyMap& properties,
                                                 bool log_fallback,
                                                 const char* log_tag) {
    const auto backend = resolve_backend_kind_from_properties(properties, log_fallback, log_tag);
    return std::string(backend_to_string(backend));
}

compiler::BackendTarget resolve_backend_target_from_properties(
    const ov::AnyMap& properties,
    bool log_fallback,
    const char* log_tag) {
    const auto request = get_backend_request(properties);
    const auto backend =
        resolve_backend_kind_from_properties(properties, log_fallback, log_tag);
    return resolve_backend_target_for_request(properties, request, backend);
}

ResolvedBackendInfo resolve_backend_for_properties(ov::AnyMap& properties,
                                                   bool log_fallback,
                                                   const char* log_tag) {
    const auto request = get_backend_request(properties);
    const auto backend = resolve_backend_kind_from_properties(properties, log_fallback, log_tag);
    ResolvedBackendInfo info;
    info.backend = backend;
    info.target = resolve_backend_target_for_request(properties, request, backend);
    info.backend_name = backend_to_string(backend);
    info.explicit_request = request.explicit_request;
    info.requested = request.requested;
    properties[kGfxBackendProperty] = info.backend_name;
    return info;
}

int parse_device_id(const ov::AnyMap& properties) {
    auto it = properties.find(ov::device::id.name());
    if (it == properties.end()) {
        return 0;
    }
    try {
        int device_id = 0;
        if (it->second.is<std::string>()) {
            device_id = std::stoi(it->second.as<std::string>());
        } else if (it->second.is<int>()) {
            device_id = it->second.as<int>();
        } else {
            OPENVINO_THROW("GFX: device id must be int or string");
        }
        if (device_id < 0) {
            OPENVINO_THROW("GFX: device id must be non-negative, got ", device_id);
        }
        return device_id;
    } catch (const std::exception& ex) {
        OPENVINO_THROW("GFX: failed to parse device id: ", ex.what());
    }
}

RemoteContextParams normalize_remote_context_params(const ov::AnyMap& remote_properties) {
    RemoteContextParams params;
    params.merged = remote_properties;
    const auto resolved =
        resolve_backend_for_properties(params.merged, /*log_fallback=*/false,
                                       "RemoteContext");
    params.backend = resolved.backend;
    params.target = resolved.target;
    params.backend_name = resolved.backend_name;
    params.merged[kGfxBackendProperty] = params.backend_name;
    params.device_id = parse_device_id(params.merged);
    params.merged[ov::device::id.name()] = params.device_id;
    return params;
}

bool apply_profiling_property(const std::string& key,
                              const ov::Any& value,
                              bool& enable_profiling,
                              ProfilingLevel& profiling_level,
                              bool& profiling_level_set,
                              ov::AnyMap& config) {
    if (key == ov::enable_profiling.name()) {
        enable_profiling = parse_bool_property(value, key);
        config[key] = enable_profiling;
        return true;
    }
    if (key == "PERF_COUNT") {  // legacy spelling accepted by benchmark_app
        enable_profiling = parse_bool_property(value, key);
        config[ov::enable_profiling.name()] = enable_profiling;
        config[key] = enable_profiling;
        return true;
    }
    if (key == kGfxProfilingLevelProperty) {
        profiling_level = parse_profiling_level(value);
        profiling_level_set = true;
        config[key] = value;
        return true;
    }
    return false;
}

bool is_diagnostic_f32_vendor_image_property(const std::string& key) {
    return key == kGfxDiagnosticF32MpsImageProperty;
}

bool parse_bool_property(const ov::Any& value, const std::string& key) {
    if (value.is<bool>()) {
        return value.as<bool>();
    }
    if (value.is<std::string>()) {
        auto text = ov::util::to_lower(value.as<std::string>());
        if (text == "true" || text == "1" || text == "yes" || text == "on") {
            return true;
        }
        if (text == "false" || text == "0" || text == "no" || text == "off") {
            return false;
        }
    }
    if (value.is<int>()) {
        return value.as<int>() != 0;
    }
    if (value.is<int64_t>()) {
        return value.as<int64_t>() != 0;
    }
    if (value.is<uint32_t>()) {
        return value.as<uint32_t>() != 0;
    }
    if (value.is<uint64_t>()) {
        return value.as<uint64_t>() != 0;
    }
    OPENVINO_THROW("GFX: property '", key, "' expects a boolean value");
}

ov::element::Type parse_inference_precision_property(const ov::Any& value,
                                                     const std::string& key) {
    ov::element::Type precision;
    if (value.is<ov::element::Type>()) {
        precision = value.as<ov::element::Type>();
    } else if (value.is<std::string>()) {
        const auto text = ov::util::to_lower(value.as<std::string>());
        if (text == "f16" || text == "fp16" || text == "half") {
            precision = ov::element::f16;
        } else if (text == "f32" || text == "fp32" || text == "float") {
            precision = ov::element::f32;
        } else {
            OPENVINO_THROW("GFX: property '", key,
                           "' supports only f16/fp16/half or f32/fp32/float");
        }
    } else {
        OPENVINO_THROW("GFX: property '", key,
                       "' expects ov::element::Type or string");
    }

    if (precision != ov::element::f16 && precision != ov::element::f32) {
        OPENVINO_THROW("GFX: property '", key, "' supports only f16 or f32");
    }
    return precision;
}

}  // namespace gfx_plugin
}  // namespace ov
