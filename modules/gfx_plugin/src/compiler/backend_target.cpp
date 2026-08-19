// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/backend_target.hpp"

#include <algorithm>
#include <sstream>
#include <utility>

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

GpuDeviceFamily normalize_family(GpuBackend backend, GpuDeviceFamily family) {
    OPENVINO_ASSERT(backend_known(backend),
                    "GFX: BackendTarget requires an explicit backend");
    switch (backend) {
        case GpuBackend::Metal:
            if (family == GpuDeviceFamily::Generic ||
                family == GpuDeviceFamily::Apple) {
                return GpuDeviceFamily::Apple;
            }
            OPENVINO_THROW(
                "GFX: Metal BackendTarget only accepts the Apple device family");
        case GpuBackend::OpenCL:
            if (family == GpuDeviceFamily::Apple) {
                OPENVINO_THROW(
                    "GFX: OpenCL BackendTarget cannot use the Apple device family");
            }
            return family;
        case GpuBackend::Unknown:
        default:
            OPENVINO_THROW("GFX: unsupported backend target kind");
    }
}

GpuDeviceFamily default_family_for_backend(GpuBackend backend) {
    switch (backend) {
        case GpuBackend::Metal:
            return GpuDeviceFamily::Apple;
        case GpuBackend::OpenCL:
            return GpuDeviceFamily::Generic;
        case GpuBackend::Unknown:
        default:
            return GpuDeviceFamily::Generic;
    }
}

std::string profile_name(GpuBackend backend, GpuDeviceFamily family) {
    const auto normalized_family = normalize_family(backend, family);
    switch (backend) {
        case GpuBackend::Metal:
            return "metal_apple";
        case GpuBackend::OpenCL:
            switch (normalized_family) {
                case GpuDeviceFamily::QualcommAdreno:
                    return "opencl_adreno";
                case GpuDeviceFamily::BroadcomV3D:
                    return "opencl_broadcom_v3d";
                case GpuDeviceFamily::Generic:
                default:
                    return "opencl_generic";
            }
        case GpuBackend::Unknown:
        default:
            OPENVINO_THROW("GFX: unsupported backend target kind");
    }
}

std::string default_device_name(GpuBackend backend, GpuDeviceFamily family) {
    const auto normalized_family = normalize_family(backend, family);
    switch (backend) {
        case GpuBackend::Metal:
            return "metal-default";
        case GpuBackend::OpenCL:
            switch (normalized_family) {
                case GpuDeviceFamily::QualcommAdreno:
                    return "qualcomm-adreno-opencl";
                case GpuDeviceFamily::BroadcomV3D:
                    return "broadcom-v3d-opencl";
                case GpuDeviceFamily::Generic:
                default:
                    return "opencl-default";
            }
        case GpuBackend::Unknown:
        default:
            OPENVINO_THROW("GFX: unsupported backend target kind");
    }
}

std::string default_vendor_id(GpuBackend backend, GpuDeviceFamily family) {
    const auto normalized_family = normalize_family(backend, family);
    switch (backend) {
        case GpuBackend::Metal:
            return "apple";
        case GpuBackend::OpenCL:
            switch (normalized_family) {
                case GpuDeviceFamily::QualcommAdreno:
                    return "qualcomm";
                case GpuDeviceFamily::BroadcomV3D:
                    return "broadcom";
                case GpuDeviceFamily::Generic:
                default:
                    return "opencl";
            }
        case GpuBackend::Unknown:
        default:
            OPENVINO_THROW("GFX: unsupported backend target kind");
    }
}

std::string default_driver_id(GpuBackend backend, GpuDeviceFamily family) {
    const auto normalized_family = normalize_family(backend, family);
    switch (backend) {
        case GpuBackend::Metal:
            return "metal-runtime";
        case GpuBackend::OpenCL:
            switch (normalized_family) {
                case GpuDeviceFamily::BroadcomV3D:
                    return "opencl-clvk";
                case GpuDeviceFamily::QualcommAdreno:
                case GpuDeviceFamily::Generic:
                default:
                    return "opencl-runtime";
            }
        case GpuBackend::Unknown:
        default:
            OPENVINO_THROW("GFX: unsupported backend target kind");
    }
}

std::string default_compiler_id(GpuBackend backend, GpuDeviceFamily family) {
    const auto normalized_family = normalize_family(backend, family);
    switch (backend) {
        case GpuBackend::Metal:
            return "gfx-mlir-msl-v1";
        case GpuBackend::OpenCL:
            if (normalized_family == GpuDeviceFamily::BroadcomV3D) {
                return "gfx-mlir-opencl-clspv-v1";
            }
            return "gfx-mlir-opencl-v1";
        case GpuBackend::Unknown:
        default:
            OPENVINO_THROW("GFX: unsupported backend target kind");
    }
}

std::vector<std::string> default_feature_bits(GpuBackend backend,
                                              GpuDeviceFamily family) {
    const auto normalized_family = normalize_family(backend, family);
    switch (backend) {
        case GpuBackend::Metal:
            return {"apple-gpu", "mps", "mpsgraph", "mpsrt", "msl",
                    "shared-mlir-route"};
        case GpuBackend::OpenCL: {
            std::vector<std::string> bits = {"opencl-source-kernels",
                                             "shared-mlir-route"};
            if (normalized_family == GpuDeviceFamily::QualcommAdreno) {
                bits.push_back("adreno");
                bits.push_back("mobile-gpu");
            }
            if (normalized_family == GpuDeviceFamily::BroadcomV3D) {
                bits.push_back("broadcom-v3d");
                bits.push_back("clspv");
                bits.push_back("clvk");
            }
            return bits;
        }
        case GpuBackend::Unknown:
        default:
            OPENVINO_THROW("GFX: unsupported backend target kind");
    }
}

void sort_unique(std::vector<std::string>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

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
    return from_backend_device_family(backend, default_family_for_backend(backend));
}

BackendTarget BackendTarget::from_backend_device_family(GpuBackend backend,
                                                        GpuDeviceFamily family) {
    return from_backend_profile(backend, family, {}, {}, {}, {});
}

BackendTarget BackendTarget::from_backend_profile(
    GpuBackend backend, GpuDeviceFamily family, std::string device_name,
    std::string vendor_id, std::string driver_id,
    std::vector<std::string> feature_bits) {
    const auto normalized_family = normalize_family(backend, family);
    BackendTarget target;
    target.m_backend = backend;
    target.m_backend_id = backend_to_string(backend);
    target.m_runtime_api = backend_to_string(backend);
    target.m_device_family = gpu_device_family_name(normalized_family);
    target.m_device_profile = profile_name(backend, normalized_family);
    target.m_device_name =
        device_name.empty() ? default_device_name(backend, normalized_family)
                            : std::move(device_name);
    target.m_vendor_id =
        vendor_id.empty() ? default_vendor_id(backend, normalized_family)
                          : std::move(vendor_id);
    target.m_driver_id =
        driver_id.empty() ? default_driver_id(backend, normalized_family)
                          : std::move(driver_id);
    target.m_compiler_id = default_compiler_id(backend, normalized_family);
    target.m_cache_compatibility_id = "gfx-target-profile-v1";
    target.m_feature_bits = default_feature_bits(backend, normalized_family);
    target.m_feature_bits.insert(target.m_feature_bits.end(), feature_bits.begin(),
                                 feature_bits.end());
    sort_unique(target.m_feature_bits);
    return target;
}

std::string BackendTarget::fingerprint() const {
    std::ostringstream oss;
    oss << "backend=" << m_backend_id
        << "|runtime=" << m_runtime_api
        << "|family=" << m_device_family
        << "|profile=" << m_device_profile
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
        << " profile=" << m_device_profile
        << " device=" << m_device_name
        << " driver=" << m_driver_id
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
