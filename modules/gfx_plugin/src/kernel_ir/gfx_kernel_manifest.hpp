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
    bool tail_outputs = false;
    uint32_t leading_input_count = 0;
    uint32_t leading_output_count = 0;
    std::vector<GfxKernelBufferRole> roles;
};

struct GfxKernelCustomManifest {
    bool valid = false;
    std::string kernel_family;
    uint32_t kernel_family_id = 0;
    std::string entry_point;
    GfxKernelExternalBufferAbiSpec external_buffer_abi;
    uint32_t threads_per_threadgroup = 0;
    bool precompiled_binary_required = false;
};

struct GfxKernelStageManifest {
    bool valid = false;
    GfxKernelStageFamily stage_family = GfxKernelStageFamily::Unknown;
    GfxKernelBackendDomain backend_domain = GfxKernelBackendDomain::Unknown;
    GfxKernelExecutionKind execution_kind = GfxKernelExecutionKind::Unknown;
    GfxKernelStorageKind storage = GfxKernelStorageKind::Unknown;
    std::string specialization_key;
    GfxKernelCustomManifest custom_kernel;
};

inline GfxKernelExternalBufferAbiSpec make_gfx_kernel_tail_outputs_abi() {
    GfxKernelExternalBufferAbiSpec spec{};
    spec.valid = true;
    spec.tail_outputs = true;
    return spec;
}

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

inline bool is_gfx_kernel_output_role(GfxKernelBufferRole role) {
    return role == GfxKernelBufferRole::TensorOutput;
}

inline GfxKernelCustomManifest make_gfx_custom_kernel_manifest(std::string kernel_family,
                                                              uint32_t kernel_family_id,
                                                              std::string entry_point,
                                                              GfxKernelExternalBufferAbiSpec external_buffer_abi,
                                                              uint32_t threads_per_threadgroup,
                                                              bool precompiled_binary_required) {
    GfxKernelCustomManifest manifest{};
    manifest.valid = true;
    manifest.kernel_family = std::move(kernel_family);
    manifest.kernel_family_id = kernel_family_id;
    manifest.entry_point = std::move(entry_point);
    manifest.external_buffer_abi = std::move(external_buffer_abi);
    manifest.threads_per_threadgroup = threads_per_threadgroup;
    manifest.precompiled_binary_required = precompiled_binary_required;
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
