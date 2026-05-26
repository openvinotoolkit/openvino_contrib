// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <string_view>

#include "runtime/gfx_mpsrt_abi.hpp"
#include "runtime/gfx_mpsrt_kernel_manifest_adapter.hpp"
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
    MPSSdpa,
    MSLDispatch,
    Alias,
};

struct GfxMpsrtStageDesc {
    GfxMpsrtStageKind kind = GfxMpsrtStageKind::Unknown;
    GfxStageBackendDomain domain = GfxStageBackendDomain::Unknown;
    GfxMpsrtStorage input_storage = GfxMpsrtStorage::Unknown;
    GfxMpsrtStorage output_storage = GfxMpsrtStorage::Unknown;
    GfxMpsrtLayout layout = GfxMpsrtLayout::Unknown;
    GfxKernelStageManifest stage_manifest;
    std::string kernel_name;
    GfxMpsrtConv2DAbiDesc conv2d_desc{};
    GfxMpsrtGemmAbiDesc gemm_desc{};
    GfxMpsrtPool2DAbiDesc pool2d_desc{};
    GfxMpsrtResize2DAbiDesc resize2d_desc{};
    GfxMpsrtSoftmaxAbiDesc softmax_desc{};
    GfxMpsrtTopKAbiDesc topk_desc{};
    GfxMpsrtSdpaAbiDesc sdpa_desc{};
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
        case GfxMpsrtStageKind::MPSSdpa:
            return "mps_sdpa";
        case GfxMpsrtStageKind::MSLDispatch:
            return "msl_dispatch";
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
    if (name == "mps_sdpa") return GfxMpsrtStageKind::MPSSdpa;
    if (name == "msl_dispatch") return GfxMpsrtStageKind::MSLDispatch;
    if (name == "alias") return GfxMpsrtStageKind::Alias;
    return GfxMpsrtStageKind::Unknown;
}

inline GfxMpsrtStageKind gfx_mpsrt_stage_kind_from_plan(const GfxStagePlacementPlan& placement,
                                                        const std::string& stage_type) {
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
    if (stage_type == "ScaledDotProductAttention") {
        return GfxMpsrtStageKind::MPSSdpa;
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
        case GfxKernelStageFamily::AttentionSoftmax:
            return GfxMpsrtStageKind::MPSSdpa;
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
        case GfxMpsrtStageKind::MPSSdpa:
            return GfxKernelStageFamily::AttentionSoftmax;
        case GfxMpsrtStageKind::MSLDispatch:
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
            if (stage_type == "ScaledDotProductAttention") {
                return GfxKernelStageFamily::AttentionSoftmax;
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
        case GfxMpsrtStageKind::MPSSdpa:
            return "ovgfx_mpsrt_encode_sdpa";
        case GfxMpsrtStageKind::MSLDispatch:
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

inline const char* gfx_mpsrt_stage_builder_symbol(const GfxMpsrtStageDesc& stage) {
    return gfx_mpsrt_builder_symbol(stage.kind);
}

inline bool gfx_mpsrt_stage_uses_vendor_primitive(const GfxMpsrtStageDesc& stage) {
    return stage.stage_manifest.valid &&
           stage.stage_manifest.execution_kind == GfxKernelExecutionKind::VendorPrimitive;
}

inline bool gfx_mpsrt_stage_uses_custom_kernel(const GfxMpsrtStageDesc& stage) {
    return stage.stage_manifest.valid &&
           stage.stage_manifest.execution_kind == GfxKernelExecutionKind::CustomKernel;
}

inline std::string gfx_mpsrt_stage_type(const GfxMpsrtStageDesc& stage) {
    return gfx_mpsrt_stage_type_from_manifest(stage.stage_manifest);
}

inline std::string gfx_mpsrt_stage_specialization_key(const GfxMpsrtStageDesc& stage) {
    return stage.stage_manifest.valid ? stage.stage_manifest.specialization_key : std::string{};
}

inline std::string gfx_mpsrt_stage_default_kernel_name(const GfxMpsrtStageDesc& stage) {
    return gfx_mpsrt_default_kernel_name(stage.kind, gfx_mpsrt_stage_type(stage));
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
    desc.kernel_name = gfx_mpsrt_default_kernel_name(desc.kind, stage_type);
    if (desc.kind != GfxMpsrtStageKind::Unknown &&
        plan.placement.uses_vendor_primitive) {
        desc.stage_manifest = make_gfx_vendor_stage_manifest(
            gfx_kernel_stage_family_from_mpsrt_kind(desc.kind, stage_type),
            gfx_mpsrt_kernel_backend_domain_from_stage_domain(plan.placement.domain),
            gfx_mpsrt_kernel_storage_from_stage_storage(plan.placement.storage),
            plan.placement.specialization_key);
    }
    if (desc.kind == GfxMpsrtStageKind::MSLDispatch) {
        const auto manifest_entry = kernel_entry_point.empty() ? std::string_view(desc.kernel_name)
                                                               : kernel_entry_point;
        const auto custom_kernel_stage = gfx_mpsrt_resolve_custom_kernel_stage_manifest(
            stage_type,
            manifest_entry,
            plan.placement.domain,
            plan.placement.storage);
        if (custom_kernel_stage.valid) {
            desc.stage_manifest = custom_kernel_stage.stage_manifest;
            if (custom_kernel_stage.dispatch.valid &&
                !custom_kernel_stage.dispatch.entry_point.empty()) {
                desc.kernel_name = custom_kernel_stage.dispatch.entry_point;
            }
        }
    }
    if (desc.stage_manifest.valid && plan.precision.keep_fp32) {
        desc.stage_manifest.compute_precision = GfxKernelComputePrecision::Fp32;
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

inline bool gfx_mpsrt_sdpa_desc_has_non_default_fields(const GfxMpsrtSdpaAbiDesc& desc) {
    return desc.has_mask != 0 || desc.causal != 0 || desc.accumulate_fp32 != 1 ||
           desc.layout != GfxMpsrtSdpaLayoutNativeBHND || desc.scale != 1.0f;
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
    key += gfx_mpsrt_stage_type(desc);
    key += "|";
    key += gfx_mpsrt_stage_specialization_key(desc);
    if (desc.stage_manifest.valid &&
        desc.stage_manifest.compute_precision == GfxKernelComputePrecision::Fp32) {
        key += "|precision:fp32";
    }
    const auto dispatch = gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(
        desc.stage_manifest.custom_kernel);
    if (desc.kind == GfxMpsrtStageKind::MSLDispatch && dispatch.valid) {
        const auto& dispatch_policy = desc.stage_manifest.custom_kernel.dispatch_policy;
        key += "|dispatch:";
        key += dispatch.kernel_family.empty() ? "unknown" : dispatch.kernel_family;
        key += ":";
        key += dispatch.entry_point.empty() ? desc.kernel_name : dispatch.entry_point;
        if (dispatch_policy.valid) {
            key += ":";
            key += gfx_kernel_dispatch_grid_name(dispatch_policy.grid);
        }
        key += ":tg";
        key += std::to_string(dispatch.threads_per_threadgroup);
        key += dispatch.precompiled_binary_required ? ":metallib" : ":source";
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
    if (desc.kind == GfxMpsrtStageKind::MPSSdpa &&
        gfx_mpsrt_sdpa_desc_has_non_default_fields(desc.sdpa_desc)) {
        key += "|sdpa:";
        key += desc.sdpa_desc.has_mask != 0 ? "mask" : "nomask";
        key += ":layout";
        key += std::to_string(desc.sdpa_desc.layout);
        if (desc.sdpa_desc.causal != 0) {
            key += ":causal";
        }
        key += ":scale";
        key += std::to_string(desc.sdpa_desc.scale);
        if (desc.sdpa_desc.accumulate_fp32 != 1) {
            key += ":acc";
            key += std::to_string(desc.sdpa_desc.accumulate_fp32);
        }
    }
    return key;
}

}  // namespace gfx_plugin
}  // namespace ov
