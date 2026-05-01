// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <string_view>

#include "runtime/gfx_mpsrt_abi.hpp"
#include "runtime/gfx_msl_kernel_manifest.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
namespace gfx_plugin {

enum class GfxMpsrtStageKind {
    Unknown,
    MPSConv2D,
    MPSGroupConv2D,
    MPSPool2D,
    MPSResize2D,
    MPSGemm,
    MPSSoftmax,
    MPSTopK,
    MSLDispatch,
    SPIRVDispatch,
    Alias,
};

struct GfxMpsrtStageDesc {
    GfxMpsrtStageKind kind = GfxMpsrtStageKind::Unknown;
    GfxStageBackendDomain domain = GfxStageBackendDomain::Unknown;
    GfxMpsrtStorage input_storage = GfxMpsrtStorage::Unknown;
    GfxMpsrtStorage output_storage = GfxMpsrtStorage::Unknown;
    GfxMpsrtLayout layout = GfxMpsrtLayout::Unknown;
    bool uses_vendor_primitive = false;
    bool uses_custom_kernel = false;
    std::string stage_type;
    std::string kernel_name;
    std::string builder_symbol;
    std::string specialization_key;
    uint32_t dispatch_kernel_family_id = 0;
    uint32_t dispatch_flags = GfxMpsrtMslDispatchFlagNone;
    std::string dispatch_kernel_family;
    std::string dispatch_entry_point;
    uint32_t dispatch_threads_per_threadgroup = 0;
    bool dispatch_precompiled_kernel_required = false;
};

inline const char* gfx_mpsrt_stage_kind_name(GfxMpsrtStageKind kind) {
    switch (kind) {
        case GfxMpsrtStageKind::MPSConv2D:
            return "mps_conv2d";
        case GfxMpsrtStageKind::MPSGroupConv2D:
            return "mps_group_conv2d";
        case GfxMpsrtStageKind::MPSPool2D:
            return "mps_pool2d";
        case GfxMpsrtStageKind::MPSResize2D:
            return "mps_resize2d";
        case GfxMpsrtStageKind::MPSGemm:
            return "mps_gemm";
        case GfxMpsrtStageKind::MPSSoftmax:
            return "mps_softmax";
        case GfxMpsrtStageKind::MPSTopK:
            return "mps_topk";
        case GfxMpsrtStageKind::MSLDispatch:
            return "msl_dispatch";
        case GfxMpsrtStageKind::SPIRVDispatch:
            return "spirv_dispatch";
        case GfxMpsrtStageKind::Alias:
            return "alias";
        case GfxMpsrtStageKind::Unknown:
        default:
            return "unknown";
    }
}

inline GfxMpsrtStageKind gfx_mpsrt_stage_kind_from_name(std::string_view name) {
    if (name == "mps_conv2d") return GfxMpsrtStageKind::MPSConv2D;
    if (name == "mps_group_conv2d") return GfxMpsrtStageKind::MPSGroupConv2D;
    if (name == "mps_pool2d") return GfxMpsrtStageKind::MPSPool2D;
    if (name == "mps_resize2d") return GfxMpsrtStageKind::MPSResize2D;
    if (name == "mps_gemm") return GfxMpsrtStageKind::MPSGemm;
    if (name == "mps_softmax") return GfxMpsrtStageKind::MPSSoftmax;
    if (name == "mps_topk") return GfxMpsrtStageKind::MPSTopK;
    if (name == "msl_dispatch") return GfxMpsrtStageKind::MSLDispatch;
    if (name == "spirv_dispatch") return GfxMpsrtStageKind::SPIRVDispatch;
    if (name == "alias") return GfxMpsrtStageKind::Alias;
    return GfxMpsrtStageKind::Unknown;
}

inline GfxMpsrtStageKind gfx_mpsrt_stage_kind_from_plan(const GfxStagePlacementPlan& placement,
                                                        const std::string& stage_type) {
    if (placement.domain == GfxStageBackendDomain::Spirv) {
        return GfxMpsrtStageKind::SPIRVDispatch;
    }
    if (placement.domain == GfxStageBackendDomain::AppleMsl) {
        return GfxMpsrtStageKind::MSLDispatch;
    }
    if (placement.domain != GfxStageBackendDomain::AppleMps) {
        return GfxMpsrtStageKind::Unknown;
    }

    if (stage_type == "Convolution") {
        return GfxMpsrtStageKind::MPSConv2D;
    }
    if (stage_type == "GroupConvolution") {
        return GfxMpsrtStageKind::MPSGroupConv2D;
    }
    if (stage_type == "MaxPool" || stage_type == "AvgPool") {
        return GfxMpsrtStageKind::MPSPool2D;
    }
    if (stage_type == "Interpolate") {
        return GfxMpsrtStageKind::MPSResize2D;
    }
    if (stage_type == "MatMul") {
        return GfxMpsrtStageKind::MPSGemm;
    }
    if (stage_type == "Softmax" || stage_type == "LogSoftmax") {
        return GfxMpsrtStageKind::MPSSoftmax;
    }
    if (stage_type == "TopK") {
        return GfxMpsrtStageKind::MPSTopK;
    }
    return GfxMpsrtStageKind::Unknown;
}

inline GfxMpsrtLayout gfx_mpsrt_stage_layout_for_storage(GfxMpsrtStorage storage) {
    switch (storage) {
        case GfxMpsrtStorage::Image:
            return GfxMpsrtLayout::NHWC4;
        case GfxMpsrtStorage::Matrix:
        case GfxMpsrtStorage::NDArray:
            return GfxMpsrtLayout::RowMajor;
        case GfxMpsrtStorage::Alias:
        case GfxMpsrtStorage::Buffer:
            return GfxMpsrtLayout::Linear;
        case GfxMpsrtStorage::Unknown:
        default:
            return GfxMpsrtLayout::Unknown;
    }
}

inline std::string gfx_mpsrt_default_kernel_name(GfxMpsrtStageKind kind, const std::string& stage_type) {
    switch (kind) {
        case GfxMpsrtStageKind::MSLDispatch:
        case GfxMpsrtStageKind::SPIRVDispatch:
            return stage_type;
        default:
            return gfx_mpsrt_stage_kind_name(kind);
    }
}

inline const char* gfx_mpsrt_builder_symbol(GfxMpsrtStageKind kind) {
    switch (kind) {
        case GfxMpsrtStageKind::MPSConv2D:
        case GfxMpsrtStageKind::MPSGroupConv2D:
            return "ovgfx_mpsrt_encode_conv2d";
        case GfxMpsrtStageKind::MPSPool2D:
            return "ovgfx_mpsrt_encode_pool2d";
        case GfxMpsrtStageKind::MPSResize2D:
            return "ovgfx_mpsrt_encode_resize2d";
        case GfxMpsrtStageKind::MPSGemm:
            return "ovgfx_mpsrt_encode_gemm";
        case GfxMpsrtStageKind::MPSSoftmax:
            return "ovgfx_mpsrt_encode_softmax";
        case GfxMpsrtStageKind::MPSTopK:
            return "ovgfx_mpsrt_encode_topk";
        case GfxMpsrtStageKind::MSLDispatch:
        case GfxMpsrtStageKind::SPIRVDispatch:
            return "ovgfx_mpsrt_encode_dispatch";
        case GfxMpsrtStageKind::Alias:
            return "ovgfx_mpsrt_encode_alias";
        case GfxMpsrtStageKind::Unknown:
        default:
            return "";
    }
}

inline bool gfx_mpsrt_stage_has_builder_symbol(GfxMpsrtStageKind kind) {
    return gfx_mpsrt_builder_symbol(kind)[0] != '\0';
}

inline GfxMpsrtStageDesc gfx_mpsrt_make_stage_desc(const GfxStageOptimizationPlan& plan,
                                                   const std::string& stage_type) {
    GfxMpsrtStageDesc desc{};
    desc.kind = gfx_mpsrt_stage_kind_from_plan(plan.placement, stage_type);
    desc.domain = plan.placement.domain;
    desc.input_storage = gfx_mpsrt_storage_from_stage_storage(plan.placement.storage);
    desc.output_storage = desc.input_storage;
    desc.layout = gfx_mpsrt_stage_layout_for_storage(desc.output_storage);
    desc.uses_vendor_primitive = plan.placement.uses_vendor_primitive;
    desc.uses_custom_kernel = plan.placement.uses_custom_kernel;
    desc.stage_type = stage_type;
    desc.kernel_name = gfx_mpsrt_default_kernel_name(desc.kind, stage_type);
    desc.builder_symbol = gfx_mpsrt_builder_symbol(desc.kind);
    desc.specialization_key = plan.placement.specialization_key;
    if (desc.kind == GfxMpsrtStageKind::MSLDispatch) {
        const auto msl_plan = make_msl_kernel_plan(stage_type, desc.kernel_name);
        if (msl_plan.valid) {
            desc.dispatch_kernel_family = msl_plan.family_name;
            desc.dispatch_entry_point = msl_plan.required_entry_point;
            desc.dispatch_kernel_family_id = msl_plan.abi_kernel_family;
            desc.dispatch_flags = msl_plan.dispatch_flags;
            desc.dispatch_threads_per_threadgroup = msl_plan.threads_per_threadgroup;
            desc.dispatch_precompiled_kernel_required = msl_plan.precompiled_metallib_required;
        }
    }
    return desc;
}

inline std::string gfx_mpsrt_stage_record_key(const GfxMpsrtStageDesc& desc) {
    std::string key = gfx_mpsrt_stage_kind_name(desc.kind);
    key += "|";
    key += gfx_stage_backend_domain_name(desc.domain);
    key += "|";
    key += gfx_mpsrt_storage_name(desc.input_storage);
    key += "|";
    key += gfx_mpsrt_storage_name(desc.output_storage);
    key += "|";
    key += gfx_mpsrt_layout_name(desc.layout);
    key += "|";
    key += desc.stage_type;
    key += "|";
    key += desc.specialization_key;
    if (desc.kind == GfxMpsrtStageKind::MSLDispatch &&
        (!desc.dispatch_kernel_family.empty() || !desc.dispatch_entry_point.empty())) {
        key += "|dispatch:";
        key += desc.dispatch_kernel_family.empty() ? "unknown" : desc.dispatch_kernel_family;
        key += ":";
        key += desc.dispatch_entry_point.empty() ? desc.kernel_name : desc.dispatch_entry_point;
        key += ":tg";
        key += std::to_string(desc.dispatch_threads_per_threadgroup);
        key += desc.dispatch_precompiled_kernel_required ? ":metallib" : ":source";
    }
    return key;
}

}  // namespace gfx_plugin
}  // namespace ov
