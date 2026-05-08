// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <string_view>

#include "runtime/gfx_mpsrt_abi.hpp"
#include "runtime/gfx_mpsrt_kernel_manifest_adapter.hpp"
#include "kernel_ir/gfx_custom_kernel_families.hpp"
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
    GfxKernelStageManifest stage_manifest;
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
    GfxMpsrtConv2DAbiDesc conv2d_desc{};
    GfxMpsrtGemmAbiDesc gemm_desc{};
    GfxMpsrtPool2DAbiDesc pool2d_desc{};
    GfxMpsrtResize2DAbiDesc resize2d_desc{};
    GfxMpsrtSoftmaxAbiDesc softmax_desc{};
    GfxMpsrtTopKAbiDesc topk_desc{};
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

inline GfxMpsrtStageKind gfx_mpsrt_stage_kind_from_manifest(const GfxKernelStageManifest& manifest) {
    if (!manifest.valid) {
        return GfxMpsrtStageKind::Unknown;
    }
    if (manifest.execution_kind == GfxKernelExecutionKind::CustomKernel) {
        if (manifest.backend_domain == GfxKernelBackendDomain::AppleMsl) {
            return GfxMpsrtStageKind::MSLDispatch;
        }
        if (manifest.backend_domain == GfxKernelBackendDomain::Spirv) {
            return GfxMpsrtStageKind::SPIRVDispatch;
        }
        return GfxMpsrtStageKind::Unknown;
    }
    if (manifest.execution_kind != GfxKernelExecutionKind::VendorPrimitive ||
        manifest.backend_domain != GfxKernelBackendDomain::AppleMps) {
        return GfxMpsrtStageKind::Unknown;
    }

    switch (manifest.stage_family) {
        case GfxKernelStageFamily::Convolution:
            return GfxMpsrtStageKind::MPSConv2D;
        case GfxKernelStageFamily::GroupConvolution:
            return GfxMpsrtStageKind::MPSGroupConv2D;
        case GfxKernelStageFamily::Pooling:
            return GfxMpsrtStageKind::MPSPool2D;
        case GfxKernelStageFamily::Resize:
            return GfxMpsrtStageKind::MPSResize2D;
        case GfxKernelStageFamily::Gemm:
            return GfxMpsrtStageKind::MPSGemm;
        case GfxKernelStageFamily::Softmax:
            return GfxMpsrtStageKind::MPSSoftmax;
        case GfxKernelStageFamily::TopK:
            return GfxMpsrtStageKind::MPSTopK;
        default:
            return GfxMpsrtStageKind::Unknown;
    }
}

inline std::string gfx_mpsrt_stage_type_from_manifest(const GfxKernelStageManifest& manifest) {
    if (!manifest.valid) {
        return {};
    }
    const auto delimiter = manifest.specialization_key.rfind(':');
    if (delimiter != std::string::npos && delimiter + 1 < manifest.specialization_key.size()) {
        return manifest.specialization_key.substr(delimiter + 1);
    }

    switch (manifest.stage_family) {
        case GfxKernelStageFamily::Convolution:
            return "Convolution";
        case GfxKernelStageFamily::GroupConvolution:
            return "GroupConvolution";
        case GfxKernelStageFamily::Pooling:
            return "Pooling";
        case GfxKernelStageFamily::Resize:
            return "Interpolate";
        case GfxKernelStageFamily::Gemm:
            return "MatMul";
        case GfxKernelStageFamily::Softmax:
            return "Softmax";
        case GfxKernelStageFamily::TopK:
            return "TopK";
        case GfxKernelStageFamily::Eltwise:
            return "Eltwise";
        case GfxKernelStageFamily::Transpose:
            return "Transpose";
        case GfxKernelStageFamily::ConcatSplit:
            return "ConcatSplit";
        case GfxKernelStageFamily::GatherScatter:
            return "GatherScatter";
        case GfxKernelStageFamily::RmsnormRope:
            return "RmsnormRope";
        case GfxKernelStageFamily::AttentionSoftmax:
            return "AttentionSoftmax";
        case GfxKernelStageFamily::KvCache:
            return "KvCache";
        case GfxKernelStageFamily::Conv3D:
            return "Conv3D";
        case GfxKernelStageFamily::Reduction:
            return "Reduction";
        case GfxKernelStageFamily::Layout:
            return "Layout";
        case GfxKernelStageFamily::Convert:
            return "Convert";
        case GfxKernelStageFamily::Unknown:
        default:
            return {};
    }
}

inline GfxKernelBackendDomain gfx_kernel_backend_domain_from_stage_domain(GfxStageBackendDomain domain) {
    switch (domain) {
        case GfxStageBackendDomain::AppleMps:
            return GfxKernelBackendDomain::AppleMps;
        case GfxStageBackendDomain::AppleMsl:
            return GfxKernelBackendDomain::AppleMsl;
        case GfxStageBackendDomain::Spirv:
            return GfxKernelBackendDomain::Spirv;
        case GfxStageBackendDomain::Unknown:
        default:
            return GfxKernelBackendDomain::Unknown;
    }
}

inline GfxKernelStorageKind gfx_kernel_storage_from_stage_storage(GfxStageStorageKind storage) {
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

inline GfxKernelStageFamily gfx_kernel_stage_family_from_mpsrt_kind(GfxMpsrtStageKind kind,
                                                                    const std::string& stage_type) {
    switch (kind) {
        case GfxMpsrtStageKind::MPSConv2D:
            return GfxKernelStageFamily::Convolution;
        case GfxMpsrtStageKind::MPSGroupConv2D:
            return GfxKernelStageFamily::GroupConvolution;
        case GfxMpsrtStageKind::MPSPool2D:
            return GfxKernelStageFamily::Pooling;
        case GfxMpsrtStageKind::MPSResize2D:
            return GfxKernelStageFamily::Resize;
        case GfxMpsrtStageKind::MPSGemm:
            return GfxKernelStageFamily::Gemm;
        case GfxMpsrtStageKind::MPSSoftmax:
            return GfxKernelStageFamily::Softmax;
        case GfxMpsrtStageKind::MPSTopK:
            return GfxKernelStageFamily::TopK;
        case GfxMpsrtStageKind::MSLDispatch:
        case GfxMpsrtStageKind::SPIRVDispatch:
            if (stage_type == "Convolution") {
                return GfxKernelStageFamily::Convolution;
            }
            if (stage_type == "GroupConvolution") {
                return GfxKernelStageFamily::GroupConvolution;
            }
            if (stage_type == "MatMul") {
                return GfxKernelStageFamily::Gemm;
            }
            if (stage_type == "Softmax" || stage_type == "LogSoftmax") {
                return GfxKernelStageFamily::Softmax;
            }
            if (stage_type == "TopK") {
                return GfxKernelStageFamily::TopK;
            }
            return GfxKernelStageFamily::Unknown;
        case GfxMpsrtStageKind::Alias:
        case GfxMpsrtStageKind::Unknown:
        default:
            return GfxKernelStageFamily::Unknown;
    }
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
                                                   const std::string& stage_type,
                                                   std::string_view kernel_entry_point = {}) {
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
    if (desc.kind != GfxMpsrtStageKind::Unknown &&
        plan.placement.uses_vendor_primitive) {
        desc.stage_manifest = make_gfx_vendor_stage_manifest(
            gfx_kernel_stage_family_from_mpsrt_kind(desc.kind, stage_type),
            gfx_kernel_backend_domain_from_stage_domain(plan.placement.domain),
            gfx_kernel_storage_from_stage_storage(plan.placement.storage),
            desc.specialization_key);
    }
    if (desc.kind == GfxMpsrtStageKind::MSLDispatch) {
        const auto manifest_entry = kernel_entry_point.empty() ? std::string_view(desc.kernel_name)
                                                               : kernel_entry_point;
        const auto custom_kernel_plan = make_gfx_custom_kernel_stage_plan(stage_type, manifest_entry);
        if (custom_kernel_plan.valid) {
            desc.stage_manifest = custom_kernel_plan.stage_manifest;
            const auto dispatch =
                gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(desc.stage_manifest.custom_kernel);
            if (dispatch.valid) {
                desc.dispatch_kernel_family = dispatch.kernel_family;
                desc.dispatch_entry_point = dispatch.entry_point;
                desc.dispatch_kernel_family_id = dispatch.kernel_family_id;
                desc.dispatch_flags = dispatch.flags;
                desc.dispatch_threads_per_threadgroup = dispatch.threads_per_threadgroup;
                desc.dispatch_precompiled_kernel_required = dispatch.precompiled_binary_required;
            }
        }
    }
    return desc;
}

inline bool gfx_mpsrt_conv2d_desc_has_non_default_fields(const GfxMpsrtConv2DAbiDesc& desc) {
    return desc.groups != 1 ||
           desc.strides[0] != 1 || desc.strides[1] != 1 ||
           desc.dilations[0] != 1 || desc.dilations[1] != 1 ||
           desc.pads[0] != 0 || desc.pads[1] != 0 ||
           desc.pads[2] != 0 || desc.pads[3] != 0 ||
           desc.fused_activation != 0 ||
           desc.accumulate_fp32 != 1;
}

inline bool gfx_mpsrt_pool2d_desc_has_non_default_fields(const GfxMpsrtPool2DAbiDesc& desc) {
    return desc.is_avg != 0 ||
           desc.kernel[0] != 1 || desc.kernel[1] != 1 ||
           desc.strides[0] != 1 || desc.strides[1] != 1 ||
           desc.dilations[0] != 1 || desc.dilations[1] != 1 ||
           desc.pads[0] != 0 || desc.pads[1] != 0 ||
           desc.pads[2] != 0 || desc.pads[3] != 0 ||
           desc.exclude_pad != 0;
}

inline bool gfx_mpsrt_resize2d_desc_has_non_default_fields(const GfxMpsrtResize2DAbiDesc& desc) {
    return desc.nearest != 0 || desc.align_corners != 0 || desc.half_pixel_centers != 1;
}

inline bool gfx_mpsrt_softmax_desc_has_non_default_fields(const GfxMpsrtSoftmaxAbiDesc& desc) {
    return desc.axis != 0 || desc.log_softmax != 0;
}

inline bool gfx_mpsrt_topk_desc_has_non_default_fields(const GfxMpsrtTopKAbiDesc& desc) {
    return desc.axis != 0 || desc.k != 0 || desc.mode_max != 1 || desc.sort_type != 0;
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
        const auto& dispatch_policy = desc.stage_manifest.custom_kernel.dispatch_policy;
        key += "|dispatch:";
        key += desc.dispatch_kernel_family.empty() ? "unknown" : desc.dispatch_kernel_family;
        key += ":";
        key += desc.dispatch_entry_point.empty() ? desc.kernel_name : desc.dispatch_entry_point;
        if (dispatch_policy.valid) {
            key += ":";
            key += gfx_kernel_dispatch_grid_name(dispatch_policy.grid);
        }
        key += ":tg";
        key += std::to_string(desc.dispatch_threads_per_threadgroup);
        key += desc.dispatch_precompiled_kernel_required ? ":metallib" : ":source";
    }
    if (desc.kind == GfxMpsrtStageKind::MPSGemm &&
        (desc.gemm_desc.transpose_lhs != 0 || desc.gemm_desc.transpose_rhs != 0 ||
         desc.gemm_desc.alpha != 1.0f || desc.gemm_desc.beta != 0.0f)) {
        key += "|gemm:ta";
        key += std::to_string(desc.gemm_desc.transpose_lhs);
        key += ":tb";
        key += std::to_string(desc.gemm_desc.transpose_rhs);
        key += ":alpha";
        key += std::to_string(desc.gemm_desc.alpha);
        key += ":beta";
        key += std::to_string(desc.gemm_desc.beta);
    }
    if ((desc.kind == GfxMpsrtStageKind::MPSConv2D ||
         desc.kind == GfxMpsrtStageKind::MPSGroupConv2D) &&
        gfx_mpsrt_conv2d_desc_has_non_default_fields(desc.conv2d_desc)) {
        key += "|conv2d:g";
        key += std::to_string(desc.conv2d_desc.groups);
        key += ":s";
        key += std::to_string(desc.conv2d_desc.strides[0]);
        key += "x";
        key += std::to_string(desc.conv2d_desc.strides[1]);
        key += ":d";
        key += std::to_string(desc.conv2d_desc.dilations[0]);
        key += "x";
        key += std::to_string(desc.conv2d_desc.dilations[1]);
        key += ":p";
        key += std::to_string(desc.conv2d_desc.pads[0]);
        key += ",";
        key += std::to_string(desc.conv2d_desc.pads[1]);
        key += ",";
        key += std::to_string(desc.conv2d_desc.pads[2]);
        key += ",";
        key += std::to_string(desc.conv2d_desc.pads[3]);
        if (desc.conv2d_desc.fused_activation != 0) {
            key += ":act";
            key += std::to_string(desc.conv2d_desc.fused_activation);
        }
        if (desc.conv2d_desc.accumulate_fp32 != 1) {
            key += ":acc";
            key += std::to_string(desc.conv2d_desc.accumulate_fp32);
        }
    }
    if (desc.kind == GfxMpsrtStageKind::MPSPool2D &&
        gfx_mpsrt_pool2d_desc_has_non_default_fields(desc.pool2d_desc)) {
        key += "|pool2d:";
        key += desc.pool2d_desc.is_avg ? "avg" : "max";
        key += ":k";
        key += std::to_string(desc.pool2d_desc.kernel[0]);
        key += "x";
        key += std::to_string(desc.pool2d_desc.kernel[1]);
        key += ":s";
        key += std::to_string(desc.pool2d_desc.strides[0]);
        key += "x";
        key += std::to_string(desc.pool2d_desc.strides[1]);
        key += ":d";
        key += std::to_string(desc.pool2d_desc.dilations[0]);
        key += "x";
        key += std::to_string(desc.pool2d_desc.dilations[1]);
        key += ":p";
        key += std::to_string(desc.pool2d_desc.pads[0]);
        key += ",";
        key += std::to_string(desc.pool2d_desc.pads[1]);
        key += ",";
        key += std::to_string(desc.pool2d_desc.pads[2]);
        key += ",";
        key += std::to_string(desc.pool2d_desc.pads[3]);
        if (desc.pool2d_desc.exclude_pad != 0) {
            key += ":exclude_pad";
        }
    }
    if (desc.kind == GfxMpsrtStageKind::MPSResize2D &&
        gfx_mpsrt_resize2d_desc_has_non_default_fields(desc.resize2d_desc)) {
        key += "|resize2d:";
        key += desc.resize2d_desc.nearest != 0 ? "nearest" : "bilinear";
        key += desc.resize2d_desc.align_corners != 0 ? ":align_corners" : "";
        key += desc.resize2d_desc.half_pixel_centers != 0 ? ":half_pixel" : "";
    }
    if (desc.kind == GfxMpsrtStageKind::MPSSoftmax &&
        gfx_mpsrt_softmax_desc_has_non_default_fields(desc.softmax_desc)) {
        key += "|softmax:axis";
        key += std::to_string(desc.softmax_desc.axis);
        if (desc.softmax_desc.log_softmax != 0) {
            key += ":log";
        }
    }
    if (desc.kind == GfxMpsrtStageKind::MPSTopK &&
        gfx_mpsrt_topk_desc_has_non_default_fields(desc.topk_desc)) {
        key += "|topk:axis";
        key += std::to_string(desc.topk_desc.axis);
        key += ":k";
        key += std::to_string(desc.topk_desc.k);
        key += desc.topk_desc.mode_max != 0 ? ":max" : ":min";
        key += ":sort";
        key += std::to_string(desc.topk_desc.sort_type);
    }
    return key;
}

}  // namespace gfx_plugin
}  // namespace ov
