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
    if (request.explicit_request) {
        OPENVINO_THROW("GFX: backend '", request.requested, "' is not available on this platform");
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
    const auto request = get_backend_request(params.merged);
    if (request.explicit_request && !backend_supported(request.kind)) {
        OPENVINO_THROW("GFX: backend '", request.requested,
                       "' is not available for remote context");
    }
    GpuBackend resolved_backend = request.explicit_request ? request.kind : default_backend_kind();
    if (!backend_supported(resolved_backend)) {
        const auto fallback = fallback_backend_kind();
        if (!backend_supported(fallback)) {
            OPENVINO_THROW("GFX: no supported backend available for remote context");
        }
        GFX_LOG_WARN("RemoteContext",
                     "Default backend '" << backend_to_string(resolved_backend)
                                         << "' is not available; falling back to "
                                         << backend_to_string(fallback));
        resolved_backend = fallback;
    }
    params.backend = resolved_backend;
    params.backend_name = backend_to_string(resolved_backend);
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
