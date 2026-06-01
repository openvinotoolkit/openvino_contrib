// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "runtime/gfx_backend_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

class BackendTarget final {
public:
    BackendTarget();

    static BackendTarget from_backend(GpuBackend backend);

    GpuBackend backend() const noexcept {
        return m_backend;
    }

    const std::string& backend_id() const noexcept {
        return m_backend_id;
    }

    const std::string& runtime_api() const noexcept {
        return m_runtime_api;
    }

    const std::string& device_family() const noexcept {
        return m_device_family;
    }

    const std::string& device_name() const noexcept {
        return m_device_name;
    }

    const std::string& vendor_id() const noexcept {
        return m_vendor_id;
    }

    const std::string& driver_id() const noexcept {
        return m_driver_id;
    }

    const std::string& compiler_id() const noexcept {
        return m_compiler_id;
    }

    const std::vector<std::string>& feature_bits() const noexcept {
        return m_feature_bits;
    }

    const std::string& cache_compatibility_id() const noexcept {
        return m_cache_compatibility_id;
    }

    std::string fingerprint() const;
    std::string debug_string() const;
    bool is_compatible_with_fingerprint(std::string_view fingerprint) const;

private:
    GpuBackend m_backend = GpuBackend::Unknown;
    std::string m_backend_id;
    std::string m_runtime_api;
    std::string m_device_family;
    std::string m_device_name;
    std::string m_vendor_id;
    std::string m_driver_id;
    std::string m_compiler_id;
    std::vector<std::string> m_feature_bits;
    std::string m_cache_compatibility_id;
};

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
