// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/backend_target.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

std::string join_feature_bits(const std::vector<std::string>& feature_bits) {
    std::ostringstream oss;
    for (size_t i = 0; i < feature_bits.size(); ++i) {
        if (i) {
            oss << ",";
        }
        oss << feature_bits[i];
    }
    return oss.str();
}

}  // namespace

BackendTarget::BackendTarget() = default;

BackendTarget BackendTarget::from_backend(GpuBackend backend) {
    BackendTarget target;
    target.m_backend = backend;
    target.m_backend_id = backend_to_string(backend);
    target.m_runtime_api = backend_to_string(backend);
    target.m_compiler_id = "gfx-mlir-v1";
    target.m_cache_compatibility_id = "gfx-target-v1";

    switch (backend) {
        case GpuBackend::Metal:
            target.m_device_family = "apple";
            target.m_device_name = "metal-default";
            target.m_vendor_id = "apple";
            target.m_driver_id = "metal-runtime";
            target.m_feature_bits = {"mps", "mpsrt", "mpsgraph", "msl"};
            break;
        case GpuBackend::OpenCL:
            target.m_device_family = "opencl-gpu";
            target.m_device_name = "opencl-default";
            target.m_vendor_id = "opencl";
            target.m_driver_id = "opencl-runtime";
            target.m_feature_bits = {"opencl-source-exceptions", "shared-mlir-route"};
            break;
    }
    return target;
}

std::string BackendTarget::fingerprint() const {
    std::ostringstream oss;
    oss << "backend=" << m_backend_id
        << "|runtime=" << m_runtime_api
        << "|family=" << m_device_family
        << "|device=" << m_device_name
        << "|vendor=" << m_vendor_id
        << "|driver=" << m_driver_id
        << "|compiler=" << m_compiler_id
        << "|cache=" << m_cache_compatibility_id
        << "|features=" << join_feature_bits(m_feature_bits);
    return oss.str();
}

std::string BackendTarget::debug_string() const {
    std::ostringstream oss;
    oss << m_backend_id << " runtime=" << m_runtime_api
        << " family=" << m_device_family
        << " compiler=" << m_compiler_id;
    return oss.str();
}

bool BackendTarget::is_compatible_with_fingerprint(std::string_view fingerprint) const {
    const auto current = this->fingerprint();
    return fingerprint == current;
}

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
