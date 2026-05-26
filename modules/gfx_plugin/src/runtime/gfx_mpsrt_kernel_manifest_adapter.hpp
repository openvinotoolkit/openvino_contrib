// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "kernel_ir/gfx_kernel_manifest.hpp"
#include "runtime/gfx_mpsrt_abi.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
namespace gfx_plugin {

struct GfxMpsrtCustomKernelDispatchSpec {
    bool valid = false;
    std::string kernel_family;
    std::string entry_point;
    uint32_t kernel_family_id = 0;
    uint32_t flags = GfxMpsrtMslDispatchFlagNone;
    uint32_t threads_per_threadgroup = 0;
    bool precompiled_binary_required = false;
};

struct GfxMpsrtResolvedCustomKernelStage {
    bool valid = false;
    GfxKernelStageManifest stage_manifest;
    GfxMpsrtCustomKernelDispatchSpec dispatch;
};

inline GfxKernelBackendDomain gfx_mpsrt_kernel_backend_domain_from_stage_domain(
    GfxStageBackendDomain domain) {
    switch (domain) {
        case GfxStageBackendDomain::AppleMps:
            return GfxKernelBackendDomain::AppleMps;
        case GfxStageBackendDomain::AppleMsl:
            return GfxKernelBackendDomain::AppleMsl;
        case GfxStageBackendDomain::OpenCl:
            return GfxKernelBackendDomain::OpenCl;
        case GfxStageBackendDomain::Unknown:
        default:
            return GfxKernelBackendDomain::Unknown;
    }
}

inline GfxKernelStorageKind gfx_mpsrt_kernel_storage_from_stage_storage(
    GfxStageStorageKind storage) {
    switch (storage) {
        case GfxStageStorageKind::Buffer:
            return GfxKernelStorageKind::Buffer;
        case GfxStageStorageKind::Image:
            return GfxKernelStorageKind::Image;
        case GfxStageStorageKind::Matrix:
            return GfxKernelStorageKind::Matrix;
        case GfxStageStorageKind::NDArray:
            return GfxKernelStorageKind::NDArray;
        case GfxStageStorageKind::Alias:
            return GfxKernelStorageKind::Alias;
        case GfxStageStorageKind::Unknown:
        default:
            return GfxKernelStorageKind::Unknown;
    }
}

inline std::string gfx_mpsrt_custom_kernel_specialization_prefix(
    GfxStageBackendDomain domain,
    GfxStageStorageKind storage) {
    std::string prefix(gfx_stage_backend_domain_name(domain));
    prefix += ":";
    prefix += gfx_stage_storage_kind_name(storage);
    prefix += ":";
    return prefix;
}

inline GfxMpsrtCustomKernelDispatchSpec gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(
    const GfxKernelCustomManifest& manifest) {
    GfxMpsrtCustomKernelDispatchSpec spec{};
    const auto& dispatch_policy = manifest.dispatch_policy;
    if (!manifest.valid ||
        manifest.kernel_family.empty() ||
        manifest.entry_point.empty() ||
        manifest.kernel_family_id == 0 ||
        !dispatch_policy.valid ||
        dispatch_policy.threads_per_threadgroup == 0) {
        return spec;
    }

    spec.valid = true;
    spec.kernel_family = manifest.kernel_family;
    spec.entry_point = manifest.entry_point;
    spec.kernel_family_id = manifest.kernel_family_id;
    spec.threads_per_threadgroup = dispatch_policy.threads_per_threadgroup;
    spec.precompiled_binary_required = dispatch_policy.precompiled_binary_required;
    if (dispatch_policy.precompiled_binary_required) {
        spec.flags |= GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired;
    }
    return spec;
}

inline GfxMpsrtResolvedCustomKernelStage gfx_mpsrt_resolve_custom_kernel_stage_manifest(
    std::string_view stage_type,
    std::string_view entry_point,
    GfxStageBackendDomain domain,
    GfxStageStorageKind storage) {
    GfxMpsrtResolvedCustomKernelStage resolved{};
    const auto backend_domain = gfx_mpsrt_kernel_backend_domain_from_stage_domain(domain);
    const auto kernel_storage = gfx_mpsrt_kernel_storage_from_stage_storage(storage);
    if (backend_domain == GfxKernelBackendDomain::Unknown ||
        kernel_storage == GfxKernelStorageKind::Unknown) {
        return resolved;
    }

    const auto custom_kernel_plan = make_gfx_custom_kernel_stage_plan(
        stage_type,
        entry_point,
        backend_domain,
        kernel_storage,
        gfx_mpsrt_custom_kernel_specialization_prefix(domain, storage));
    if (!custom_kernel_plan.valid) {
        return resolved;
    }

    resolved.stage_manifest = custom_kernel_plan.stage_manifest;
    resolved.dispatch = gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(
        resolved.stage_manifest.custom_kernel);
    resolved.valid = resolved.stage_manifest.valid;
    return resolved;
}

inline GfxMpsrtExternalBufferRole gfx_mpsrt_external_buffer_role_from_kernel_role(
    GfxKernelBufferRole role) {
    switch (role) {
        case GfxKernelBufferRole::TensorInput:
            return GfxMpsrtExternalBufferRole::TensorInput;
        case GfxKernelBufferRole::TensorOutput:
            return GfxMpsrtExternalBufferRole::TensorOutput;
        case GfxKernelBufferRole::RuntimeParams:
            return GfxMpsrtExternalBufferRole::RuntimeParams;
        case GfxKernelBufferRole::ConstTensor:
            return GfxMpsrtExternalBufferRole::ConstBuffer;
        case GfxKernelBufferRole::ScalarParam:
        case GfxKernelBufferRole::Unknown:
        default:
            return GfxMpsrtExternalBufferRole::Unknown;
    }
}

inline bool gfx_mpsrt_is_valid_external_buffer_role(GfxMpsrtExternalBufferRole role) {
    switch (role) {
        case GfxMpsrtExternalBufferRole::TensorInput:
        case GfxMpsrtExternalBufferRole::TensorOutput:
        case GfxMpsrtExternalBufferRole::ConstBuffer:
        case GfxMpsrtExternalBufferRole::RuntimeParams:
        case GfxMpsrtExternalBufferRole::Metadata:
            return true;
        case GfxMpsrtExternalBufferRole::Unknown:
        default:
            return false;
    }
}

inline bool gfx_mpsrt_is_external_output_buffer_role(GfxMpsrtExternalBufferRole role) {
    return role == GfxMpsrtExternalBufferRole::TensorOutput;
}

inline std::vector<GfxMpsrtExternalBufferRole> gfx_mpsrt_external_buffer_roles_from_kernel_roles(
    const std::vector<GfxKernelBufferRole>& roles) {
    std::vector<GfxMpsrtExternalBufferRole> mpsrt_roles;
    mpsrt_roles.reserve(roles.size());
    for (const auto role : roles) {
        if (is_gfx_kernel_scalar_role(role)) {
            continue;
        }
        const auto mpsrt_role = gfx_mpsrt_external_buffer_role_from_kernel_role(role);
        if (mpsrt_role == GfxMpsrtExternalBufferRole::Unknown) {
            return {};
        }
        mpsrt_roles.push_back(mpsrt_role);
    }
    return mpsrt_roles;
}

inline uint32_t gfx_mpsrt_count_external_output_roles(
    const std::vector<GfxMpsrtExternalBufferRole>& roles) {
    uint32_t count = 0;
    for (const auto role : roles) {
        if (role == GfxMpsrtExternalBufferRole::TensorOutput) {
            ++count;
        }
    }
    return count;
}

inline std::vector<GfxMpsrtValue> gfx_mpsrt_kernel_buffer_order_from_external_roles(
    const std::vector<GfxMpsrtExternalBufferRole>& roles,
    const std::vector<GfxMpsrtValue>& input_values,
    const std::vector<GfxMpsrtValue>& output_values) {
    if (roles.empty()) {
        return {};
    }

    std::vector<GfxMpsrtValue> order;
    order.reserve(roles.size());
    size_t next_input = 0;
    size_t next_output = 0;
    for (const auto role : roles) {
        switch (role) {
            case GfxMpsrtExternalBufferRole::TensorInput:
            case GfxMpsrtExternalBufferRole::ConstBuffer:
            case GfxMpsrtExternalBufferRole::RuntimeParams:
            case GfxMpsrtExternalBufferRole::Metadata:
                if (next_input >= input_values.size()) {
                    break;
                }
                order.push_back(input_values[next_input++]);
                break;
            case GfxMpsrtExternalBufferRole::TensorOutput:
                if (next_output >= output_values.size()) {
                    break;
                }
                order.push_back(output_values[next_output++]);
                break;
            case GfxMpsrtExternalBufferRole::Unknown:
            default:
                return {};
        }
    }
    if (next_input != input_values.size() || next_output != output_values.size()) {
        return {};
    }
    return order;
}

inline std::vector<GfxMpsrtValue> gfx_mpsrt_kernel_buffer_order_from_external_values(
    const std::vector<GfxMpsrtExternalBufferRole>& roles,
    const std::vector<GfxMpsrtValue>& external_values) {
    if (roles.empty() || roles.size() != external_values.size()) {
        return {};
    }
    for (const auto role : roles) {
        if (!gfx_mpsrt_is_valid_external_buffer_role(role)) {
            return {};
        }
    }
    return external_values;
}

inline std::vector<GfxMpsrtExternalBufferRole> gfx_mpsrt_external_buffer_roles_from_direct_io_spec(
    const GfxKernelExternalBufferAbiSpec& spec) {
    const uint32_t direct_io_count = spec.direct_input_count + spec.direct_output_count;
    if (!spec.valid || direct_io_count == 0 || spec.direct_output_count == 0) {
        return {};
    }

    std::vector<GfxMpsrtExternalBufferRole> roles;
    roles.reserve(direct_io_count);
    roles.insert(roles.end(), spec.direct_input_count, GfxMpsrtExternalBufferRole::TensorInput);
    roles.insert(roles.end(), spec.direct_output_count, GfxMpsrtExternalBufferRole::TensorOutput);
    return roles;
}

inline std::vector<GfxMpsrtValue> gfx_mpsrt_kernel_buffer_order_from_kernel_abi(
    const GfxKernelExternalBufferAbiSpec& spec,
    const std::vector<GfxMpsrtValue>& input_values,
    const std::vector<GfxMpsrtValue>& output_values) {
    if (!spec.valid) {
        return {};
    }

    std::vector<GfxMpsrtExternalBufferRole> roles;
    if (!spec.roles.empty()) {
        roles = gfx_mpsrt_external_buffer_roles_from_kernel_roles(spec.roles);
    } else if (spec.direct_input_count != 0 || spec.direct_output_count != 0) {
        roles = gfx_mpsrt_external_buffer_roles_from_direct_io_spec(spec);
    }

    return gfx_mpsrt_kernel_buffer_order_from_external_roles(roles, input_values, output_values);
}

}  // namespace gfx_plugin
}  // namespace ov
