// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/gfx_plugin/properties.hpp"
#include "openvino/gfx_plugin/profiling.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_backend_utils.hpp"

namespace ov {
namespace gfx_plugin {

struct BackendRequest {
    GpuBackend kind = GpuBackend::Unknown;
    bool explicit_request = false;
    std::string requested;
};

struct ResolvedBackendInfo {
    GpuBackend backend = GpuBackend::Unknown;
    std::string backend_name;
    bool explicit_request = false;
    std::string requested;
};

struct RemoteContextParams {
    ov::AnyMap merged;
    GpuBackend backend = GpuBackend::Unknown;
    std::string backend_name;
    int device_id = 0;
};

BackendRequest get_backend_request(const ov::AnyMap& properties);

GpuBackend resolve_backend_kind_from_properties(const ov::AnyMap& properties,
                                                bool log_fallback,
                                                const char* log_tag);

std::string resolve_backend_name_from_properties(const ov::AnyMap& properties,
                                                 bool log_fallback,
                                                 const char* log_tag);

ResolvedBackendInfo resolve_backend_for_properties(ov::AnyMap& properties,
                                                   bool log_fallback,
                                                   const char* log_tag);

int parse_device_id(const ov::AnyMap& properties);

RemoteContextParams normalize_remote_context_params(const ov::AnyMap& remote_properties);

bool apply_profiling_property(const std::string& key,
                              const ov::Any& value,
                              bool& enable_profiling,
                              ProfilingLevel& profiling_level,
                              bool& profiling_level_set,
                              ov::AnyMap& config);

bool is_diagnostic_f32_vendor_image_property(const std::string& key);

bool parse_bool_property(const ov::Any& value, const std::string& key);
ov::element::Type parse_inference_precision_property(const ov::Any& value,
                                                     const std::string& key);

}  // namespace gfx_plugin
}  // namespace ov
