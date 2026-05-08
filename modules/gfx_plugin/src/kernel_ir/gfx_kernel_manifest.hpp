// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ov {
namespace gfx_plugin {

enum class GfxKernelBufferRole : uint32_t {
    Unknown = 0,
    TensorInput = 1,
    TensorOutput = 2,
    RuntimeParams = 3,
    ConstTensor = 4,
    ScalarParam = 5,
};

enum class GfxKernelBackendDomain : uint32_t {
    Unknown = 0,
    AppleMps = 1,
    AppleMsl = 2,
    Spirv = 3,
};

enum class GfxKernelExecutionKind : uint32_t {
    Unknown = 0,
    VendorPrimitive = 1,
    CustomKernel = 2,
};

enum class GfxKernelStorageKind : uint32_t {
    Unknown = 0,
    Buffer = 1,
    Image = 2,
    Matrix = 3,
    NDArray = 4,
    Alias = 5,
};

enum class GfxKernelStageFamily : uint32_t {
    Unknown = 0,
    Convolution = 1,
    GroupConvolution = 2,
    Pooling = 3,
    Resize = 4,
    Gemm = 5,
    Softmax = 6,
    TopK = 7,
    Eltwise = 8,
    Transpose = 9,
    ConcatSplit = 10,
    GatherScatter = 11,
    RmsnormRope = 12,
    AttentionSoftmax = 13,
    KvCache = 14,
    Conv3D = 15,
    Reduction = 16,
    Layout = 17,
    Convert = 18,
};

enum class GfxKernelDispatchGrid : uint32_t {
    Unknown = 0,
    Linear1D = 1,
    Tiled2D = 2,
    Tiled3D = 3,
};

inline const char* gfx_kernel_dispatch_grid_name(GfxKernelDispatchGrid grid) {
    switch (grid) {
        case GfxKernelDispatchGrid::Linear1D:
            return "linear_1d";
        case GfxKernelDispatchGrid::Tiled2D:
            return "tiled_2d";
        case GfxKernelDispatchGrid::Tiled3D:
            return "tiled_3d";
        case GfxKernelDispatchGrid::Unknown:
        default:
            return "unknown";
    }
}

inline GfxKernelDispatchGrid gfx_kernel_dispatch_grid_from_name(std::string_view name) {
    if (name == "linear_1d") return GfxKernelDispatchGrid::Linear1D;
    if (name == "tiled_2d") return GfxKernelDispatchGrid::Tiled2D;
    if (name == "tiled_3d") return GfxKernelDispatchGrid::Tiled3D;
    return GfxKernelDispatchGrid::Unknown;
}

inline const char* gfx_kernel_backend_domain_name(GfxKernelBackendDomain domain) {
    switch (domain) {
        case GfxKernelBackendDomain::AppleMps:
            return "apple_mps";
        case GfxKernelBackendDomain::AppleMsl:
            return "apple_msl";
        case GfxKernelBackendDomain::Spirv:
            return "spirv";
        case GfxKernelBackendDomain::Unknown:
        default:
            return "unknown";
    }
}

inline GfxKernelBackendDomain gfx_kernel_backend_domain_from_name(std::string_view name) {
    if (name == "apple_mps") return GfxKernelBackendDomain::AppleMps;
    if (name == "apple_msl") return GfxKernelBackendDomain::AppleMsl;
    if (name == "spirv") return GfxKernelBackendDomain::Spirv;
    return GfxKernelBackendDomain::Unknown;
}

inline const char* gfx_kernel_execution_kind_name(GfxKernelExecutionKind kind) {
    switch (kind) {
        case GfxKernelExecutionKind::VendorPrimitive:
            return "vendor_primitive";
        case GfxKernelExecutionKind::CustomKernel:
            return "custom_kernel";
        case GfxKernelExecutionKind::Unknown:
        default:
            return "unknown";
    }
}

inline GfxKernelExecutionKind gfx_kernel_execution_kind_from_name(std::string_view name) {
    if (name == "vendor_primitive") return GfxKernelExecutionKind::VendorPrimitive;
    if (name == "custom_kernel") return GfxKernelExecutionKind::CustomKernel;
    return GfxKernelExecutionKind::Unknown;
}

inline const char* gfx_kernel_storage_kind_name(GfxKernelStorageKind storage) {
    switch (storage) {
        case GfxKernelStorageKind::Buffer:
            return "buffer";
        case GfxKernelStorageKind::Image:
            return "image";
        case GfxKernelStorageKind::Matrix:
            return "matrix";
        case GfxKernelStorageKind::NDArray:
            return "ndarray";
        case GfxKernelStorageKind::Alias:
            return "alias";
        case GfxKernelStorageKind::Unknown:
        default:
            return "unknown";
    }
}

inline GfxKernelStorageKind gfx_kernel_storage_kind_from_name(std::string_view name) {
    if (name == "buffer") return GfxKernelStorageKind::Buffer;
    if (name == "image") return GfxKernelStorageKind::Image;
    if (name == "matrix") return GfxKernelStorageKind::Matrix;
    if (name == "ndarray") return GfxKernelStorageKind::NDArray;
    if (name == "alias") return GfxKernelStorageKind::Alias;
    return GfxKernelStorageKind::Unknown;
}

inline const char* gfx_kernel_stage_family_name(GfxKernelStageFamily family) {
    switch (family) {
        case GfxKernelStageFamily::Convolution:
            return "convolution";
        case GfxKernelStageFamily::GroupConvolution:
            return "group_convolution";
        case GfxKernelStageFamily::Pooling:
            return "pooling";
        case GfxKernelStageFamily::Resize:
            return "resize";
        case GfxKernelStageFamily::Gemm:
            return "gemm";
        case GfxKernelStageFamily::Softmax:
            return "softmax";
        case GfxKernelStageFamily::TopK:
            return "topk";
        case GfxKernelStageFamily::Eltwise:
            return "eltwise";
        case GfxKernelStageFamily::Transpose:
            return "transpose";
        case GfxKernelStageFamily::ConcatSplit:
            return "concat_split";
        case GfxKernelStageFamily::GatherScatter:
            return "gather_scatter";
        case GfxKernelStageFamily::RmsnormRope:
            return "rmsnorm_rope";
        case GfxKernelStageFamily::AttentionSoftmax:
            return "attention_softmax";
        case GfxKernelStageFamily::KvCache:
            return "kv_cache";
        case GfxKernelStageFamily::Conv3D:
            return "conv3d";
        case GfxKernelStageFamily::Reduction:
            return "reduction";
        case GfxKernelStageFamily::Layout:
            return "layout";
        case GfxKernelStageFamily::Convert:
            return "convert";
        case GfxKernelStageFamily::Unknown:
        default:
            return "unknown";
    }
}

inline GfxKernelStageFamily gfx_kernel_stage_family_from_name(std::string_view name) {
    if (name == "convolution") return GfxKernelStageFamily::Convolution;
    if (name == "group_convolution") return GfxKernelStageFamily::GroupConvolution;
    if (name == "pooling") return GfxKernelStageFamily::Pooling;
    if (name == "resize") return GfxKernelStageFamily::Resize;
    if (name == "gemm") return GfxKernelStageFamily::Gemm;
    if (name == "softmax") return GfxKernelStageFamily::Softmax;
    if (name == "topk") return GfxKernelStageFamily::TopK;
    if (name == "eltwise") return GfxKernelStageFamily::Eltwise;
    if (name == "transpose") return GfxKernelStageFamily::Transpose;
    if (name == "concat_split") return GfxKernelStageFamily::ConcatSplit;
    if (name == "gather_scatter") return GfxKernelStageFamily::GatherScatter;
    if (name == "rmsnorm_rope") return GfxKernelStageFamily::RmsnormRope;
    if (name == "attention_softmax") return GfxKernelStageFamily::AttentionSoftmax;
    if (name == "kv_cache") return GfxKernelStageFamily::KvCache;
    if (name == "conv3d") return GfxKernelStageFamily::Conv3D;
    if (name == "reduction") return GfxKernelStageFamily::Reduction;
    if (name == "layout") return GfxKernelStageFamily::Layout;
    if (name == "convert") return GfxKernelStageFamily::Convert;
    return GfxKernelStageFamily::Unknown;
}

struct GfxKernelExternalBufferAbiSpec {
    bool valid = false;
    uint32_t leading_input_count = 0;
    uint32_t leading_output_count = 0;
    uint32_t direct_input_count = 0;
    uint32_t direct_output_count = 0;
    std::vector<GfxKernelBufferRole> roles;
};

struct GfxKernelDispatchPolicy {
    bool valid = false;
    GfxKernelDispatchGrid grid = GfxKernelDispatchGrid::Unknown;
    uint32_t threads_per_threadgroup = 0;
    bool precompiled_binary_required = false;
};

struct GfxKernelCustomManifest {
    bool valid = false;
    std::string kernel_family;
    uint32_t kernel_family_id = 0;
    std::string entry_point;
    GfxKernelExternalBufferAbiSpec external_buffer_abi;
    GfxKernelDispatchPolicy dispatch_policy;
    std::vector<int32_t> scalar_args;
};

struct GfxKernelStageManifest {
    bool valid = false;
    GfxKernelStageFamily stage_family = GfxKernelStageFamily::Unknown;
    GfxKernelBackendDomain backend_domain = GfxKernelBackendDomain::Unknown;
    GfxKernelExecutionKind execution_kind = GfxKernelExecutionKind::Unknown;
    GfxKernelStorageKind storage = GfxKernelStorageKind::Unknown;
    std::string specialization_key;
    std::vector<GfxKernelBufferRole> semantic_input_roles;
    std::vector<GfxKernelBufferRole> semantic_output_roles;
    GfxKernelCustomManifest custom_kernel;
};

inline GfxKernelExternalBufferAbiSpec make_gfx_kernel_leading_io_params_abi(uint32_t input_count,
                                                                           uint32_t output_count) {
    GfxKernelExternalBufferAbiSpec spec{};
    spec.valid = true;
    spec.leading_input_count = input_count;
    spec.leading_output_count = output_count;
    return spec;
}

inline GfxKernelExternalBufferAbiSpec make_gfx_kernel_roles_abi(std::vector<GfxKernelBufferRole> roles) {
    GfxKernelExternalBufferAbiSpec spec{};
    spec.valid = true;
    spec.roles = std::move(roles);
    return spec;
}

inline GfxKernelExternalBufferAbiSpec make_gfx_kernel_direct_io_abi(uint32_t input_count,
                                                                    uint32_t output_count) {
    GfxKernelExternalBufferAbiSpec spec{};
    spec.valid = true;
    spec.direct_input_count = input_count;
    spec.direct_output_count = output_count;
    return spec;
}

inline std::vector<GfxKernelBufferRole> materialize_gfx_kernel_external_buffer_roles(
    const GfxKernelExternalBufferAbiSpec& abi) {
    std::vector<GfxKernelBufferRole> roles = abi.roles;
    if (roles.empty() && (abi.direct_input_count != 0 || abi.direct_output_count != 0)) {
        roles.insert(roles.end(), abi.direct_input_count, GfxKernelBufferRole::TensorInput);
        roles.insert(roles.end(), abi.direct_output_count, GfxKernelBufferRole::TensorOutput);
    }
    if (roles.empty() && (abi.leading_input_count != 0 || abi.leading_output_count != 0)) {
        roles.insert(roles.end(), abi.leading_input_count, GfxKernelBufferRole::TensorInput);
        roles.insert(roles.end(), abi.leading_output_count, GfxKernelBufferRole::TensorOutput);
    }
    return roles;
}

inline GfxKernelDispatchPolicy make_gfx_kernel_dispatch_policy(
    GfxKernelDispatchGrid grid,
    uint32_t threads_per_threadgroup,
    bool precompiled_binary_required) {
    GfxKernelDispatchPolicy policy{};
    policy.valid = grid != GfxKernelDispatchGrid::Unknown &&
                   threads_per_threadgroup != 0;
    policy.grid = grid;
    policy.threads_per_threadgroup = threads_per_threadgroup;
    policy.precompiled_binary_required = precompiled_binary_required;
    return policy;
}

inline GfxKernelDispatchPolicy make_gfx_kernel_linear_dispatch_policy(
    uint32_t threads_per_threadgroup,
    bool precompiled_binary_required) {
    return make_gfx_kernel_dispatch_policy(GfxKernelDispatchGrid::Linear1D,
                                           threads_per_threadgroup,
                                           precompiled_binary_required);
}

inline bool is_gfx_kernel_output_role(GfxKernelBufferRole role) {
    return role == GfxKernelBufferRole::TensorOutput;
}

inline bool is_gfx_kernel_scalar_role(GfxKernelBufferRole role) {
    return role == GfxKernelBufferRole::ScalarParam;
}

inline bool is_gfx_kernel_buffer_role(GfxKernelBufferRole role) {
    switch (role) {
        case GfxKernelBufferRole::TensorInput:
        case GfxKernelBufferRole::TensorOutput:
        case GfxKernelBufferRole::RuntimeParams:
        case GfxKernelBufferRole::ConstTensor:
            return true;
        case GfxKernelBufferRole::ScalarParam:
        case GfxKernelBufferRole::Unknown:
        default:
            return false;
    }
}

inline GfxKernelCustomManifest make_gfx_custom_kernel_manifest(std::string kernel_family,
                                                              uint32_t kernel_family_id,
                                                              std::string entry_point,
                                                              GfxKernelExternalBufferAbiSpec external_buffer_abi,
                                                              GfxKernelDispatchPolicy dispatch_policy) {
    GfxKernelCustomManifest manifest{};
    manifest.valid = true;
    manifest.kernel_family = std::move(kernel_family);
    manifest.kernel_family_id = kernel_family_id;
    manifest.entry_point = std::move(entry_point);
    manifest.external_buffer_abi = std::move(external_buffer_abi);
    manifest.dispatch_policy = dispatch_policy;
    return manifest;
}

inline GfxKernelStageManifest make_gfx_vendor_stage_manifest(GfxKernelStageFamily family,
                                                            GfxKernelBackendDomain domain,
                                                            GfxKernelStorageKind storage,
                                                            std::string specialization_key) {
    GfxKernelStageManifest manifest{};
    manifest.valid = true;
    manifest.stage_family = family;
    manifest.backend_domain = domain;
    manifest.execution_kind = GfxKernelExecutionKind::VendorPrimitive;
    manifest.storage = storage;
    manifest.specialization_key = std::move(specialization_key);
    return manifest;
}

inline GfxKernelStageManifest make_gfx_custom_kernel_stage_manifest(GfxKernelStageFamily family,
                                                                   GfxKernelBackendDomain domain,
                                                                   GfxKernelStorageKind storage,
                                                                   std::string specialization_key,
                                                                   GfxKernelCustomManifest custom_kernel) {
    GfxKernelStageManifest manifest{};
    manifest.valid = true;
    manifest.stage_family = family;
    manifest.backend_domain = domain;
    manifest.execution_kind = GfxKernelExecutionKind::CustomKernel;
    manifest.storage = storage;
    manifest.specialization_key = std::move(specialization_key);
    manifest.custom_kernel = std::move(custom_kernel);
    return manifest;
}

}  // namespace gfx_plugin
}  // namespace ov
