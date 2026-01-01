// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gfx_property_utils.hpp"

#include "openvino/core/except.hpp"
#include "plugin/gfx_profiling_utils.hpp"

namespace ov {
namespace gfx_plugin {

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
    if (backend_supported(request.kind)) {
        return request.kind;
    }
    const auto fallback = fallback_backend_kind();
    if (!backend_supported(fallback)) {
        OPENVINO_THROW("GFX: no supported backend available on this platform.");
    }
    if (log_fallback) {
        if (request.explicit_request) {
            GFX_LOG_WARN(log_tag,
                         "GFX_BACKEND=" << request.requested
                                        << " is not available on this platform; falling back to "
                                        << backend_to_string(fallback));
        } else {
            GFX_LOG_WARN(log_tag,
                         "Default GFX backend is not available on this platform; falling back to "
                                        << backend_to_string(fallback));
        }
    }
    return fallback;
}

std::string resolve_backend_name_from_properties(const ov::AnyMap& properties,
                                                 bool log_fallback,
                                                 const char* log_tag) {
    const auto backend = resolve_backend_kind_from_properties(properties, log_fallback, log_tag);
    return std::string(backend_to_string(backend));
}

ResolvedBackendInfo resolve_backend_for_properties(ov::AnyMap& properties,
                                                   bool log_fallback,
                                                   const char* log_tag) {
    const auto request = get_backend_request(properties);
    const auto backend = resolve_backend_kind_from_properties(properties, log_fallback, log_tag);
    ResolvedBackendInfo info;
    info.backend = backend;
    info.backend_name = backend_to_string(backend);
    info.explicit_request = request.explicit_request;
    info.requested = request.requested;
    properties[kGfxBackendProperty] = info.backend_name;
    return info;
}

int parse_device_id(const ov::AnyMap& properties) {
    int device_id = 0;
    auto it = properties.find(ov::device::id.name());
    if (it == properties.end()) {
        return device_id;
    }
    try {
        if (it->second.is<std::string>()) {
            device_id = std::stoi(it->second.as<std::string>());
        } else {
            device_id = it->second.as<int>();
        }
    } catch (...) {
        device_id = 0;
    }
    return device_id;
}

RemoteContextParams normalize_remote_context_params(const ov::AnyMap& remote_properties) {
    RemoteContextParams params;
    params.merged = remote_properties;
    params.backend = resolve_backend_kind_from_properties(params.merged,
                                                          /*log_fallback=*/true,
                                                          "RemoteContext");
    params.backend_name = backend_to_string(params.backend);
    params.merged[kGfxBackendProperty] = params.backend_name;
    params.device_id = parse_device_id(params.merged);
    return params;
}

bool apply_profiling_property(const std::string& key,
                              const ov::Any& value,
                              bool& enable_profiling,
                              ProfilingLevel& profiling_level,
                              bool& profiling_level_set,
                              ov::AnyMap& config) {
    if (key == ov::enable_profiling.name()) {
        enable_profiling = value.as<bool>();
        config[key] = enable_profiling;
        return true;
    }
    if (key == "PERF_COUNT") {  // legacy spelling accepted by benchmark_app
        enable_profiling = value.as<bool>();
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

}  // namespace gfx_plugin
}  // namespace ov
