// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kernel_ir/gfx_kernel_manifest.hpp"
#include "runtime/gfx_mpsrt_abi.hpp"

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

inline GfxMpsrtCustomKernelDispatchSpec gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(
    const GfxKernelCustomManifest& manifest) {
    GfxMpsrtCustomKernelDispatchSpec spec{};
    if (!manifest.valid ||
        manifest.kernel_family.empty() ||
        manifest.entry_point.empty() ||
        manifest.kernel_family_id == 0 ||
        manifest.threads_per_threadgroup == 0) {
        return spec;
    }

    spec.valid = true;
    spec.kernel_family = manifest.kernel_family;
    spec.entry_point = manifest.entry_point;
    spec.kernel_family_id = manifest.kernel_family_id;
    spec.threads_per_threadgroup = manifest.threads_per_threadgroup;
    spec.precompiled_binary_required = manifest.precompiled_binary_required;
    if (manifest.precompiled_binary_required) {
        spec.flags |= GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired;
    }
    return spec;
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

inline std::vector<GfxMpsrtValue> gfx_mpsrt_default_kernel_buffer_order(
    const std::vector<GfxMpsrtValue>& input_values,
    const std::vector<GfxMpsrtValue>& output_values) {
    std::vector<GfxMpsrtValue> order;
    order.reserve(input_values.size() + output_values.size());
    order.insert(order.end(), input_values.begin(), input_values.end());
    order.insert(order.end(), output_values.begin(), output_values.end());
    return order;
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
                if (next_input >= input_values.size()) {
                    return {};
                }
                order.push_back(input_values[next_input++]);
                break;
            case GfxMpsrtExternalBufferRole::TensorOutput:
                if (next_output >= output_values.size()) {
                    return {};
                }
                order.push_back(output_values[next_output++]);
                break;
            case GfxMpsrtExternalBufferRole::ConstBuffer:
            case GfxMpsrtExternalBufferRole::RuntimeParams:
            case GfxMpsrtExternalBufferRole::Metadata:
                return {};
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

inline std::vector<GfxMpsrtExternalBufferRole> gfx_mpsrt_external_buffer_roles_from_leading_io_spec(
    const GfxKernelExternalBufferAbiSpec& spec,
    uint32_t buffer_count) {
    const uint32_t structured_count = spec.leading_input_count + spec.leading_output_count;
    if (!spec.valid || structured_count == 0 || buffer_count < structured_count) {
        return {};
    }

    std::vector<GfxMpsrtExternalBufferRole> roles;
    roles.reserve(buffer_count);
    roles.insert(roles.end(), spec.leading_input_count, GfxMpsrtExternalBufferRole::TensorInput);
    roles.insert(roles.end(), spec.leading_output_count, GfxMpsrtExternalBufferRole::TensorOutput);
    roles.insert(roles.end(), buffer_count - structured_count, GfxMpsrtExternalBufferRole::RuntimeParams);
    return roles;
}

inline std::vector<GfxMpsrtExternalBufferRole> gfx_mpsrt_external_buffer_roles_from_tail_outputs(
    uint32_t buffer_count,
    uint32_t output_buffer_count) {
    if (output_buffer_count == 0 || output_buffer_count > buffer_count) {
        return {};
    }

    std::vector<GfxMpsrtExternalBufferRole> roles;
    roles.reserve(buffer_count);
    const uint32_t input_buffer_count = buffer_count - output_buffer_count;
    for (uint32_t i = 0; i < buffer_count; ++i) {
        roles.push_back(i < input_buffer_count ? GfxMpsrtExternalBufferRole::TensorInput
                                               : GfxMpsrtExternalBufferRole::TensorOutput);
    }
    return roles;
}

inline std::vector<GfxMpsrtValue> gfx_mpsrt_kernel_buffer_order_from_kernel_abi(
    const GfxKernelExternalBufferAbiSpec& spec,
    const std::vector<GfxMpsrtValue>& input_values,
    const std::vector<GfxMpsrtValue>& output_values) {
    if (!spec.valid) {
        return gfx_mpsrt_default_kernel_buffer_order(input_values, output_values);
    }

    std::vector<GfxMpsrtExternalBufferRole> roles;
    const uint32_t semantic_buffer_count = static_cast<uint32_t>(input_values.size() + output_values.size());
    if (!spec.roles.empty()) {
        roles = gfx_mpsrt_external_buffer_roles_from_kernel_roles(spec.roles);
    } else if (spec.tail_outputs) {
        roles = gfx_mpsrt_external_buffer_roles_from_tail_outputs(semantic_buffer_count,
                                                                  static_cast<uint32_t>(output_values.size()));
    } else if (spec.leading_input_count != 0 || spec.leading_output_count != 0) {
        roles = gfx_mpsrt_external_buffer_roles_from_leading_io_spec(spec, semantic_buffer_count);
    }

    auto order = gfx_mpsrt_kernel_buffer_order_from_external_roles(roles, input_values, output_values);
    if (!order.empty()) {
        return order;
    }
    return gfx_mpsrt_default_kernel_buffer_order(input_values, output_values);
}

}  // namespace gfx_plugin
}  // namespace ov
