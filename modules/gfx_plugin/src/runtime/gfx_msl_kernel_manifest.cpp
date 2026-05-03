// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_msl_kernel_manifest.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

GfxKernelStageFamily gfx_msl_stage_family_from_kernel_family(GfxMslKernelFamily family) {
    switch (family) {
        case GfxMslKernelFamily::EltwiseFusedBuffer:
            return GfxKernelStageFamily::Eltwise;
        case GfxMslKernelFamily::TransposePackND:
            return GfxKernelStageFamily::Transpose;
        case GfxMslKernelFamily::ConcatSplitGeneric:
            return GfxKernelStageFamily::ConcatSplit;
        case GfxMslKernelFamily::GatherScatterIndexed:
            return GfxKernelStageFamily::GatherScatter;
        case GfxMslKernelFamily::RmsnormRopeFused:
            return GfxKernelStageFamily::RmsnormRope;
        case GfxMslKernelFamily::MaskedSoftmaxAttention:
            return GfxKernelStageFamily::AttentionSoftmax;
        case GfxMslKernelFamily::KvCacheUpdate:
            return GfxKernelStageFamily::KvCache;
        case GfxMslKernelFamily::Conv3DDirectOrIm2col:
            return GfxKernelStageFamily::Conv3D;
        case GfxMslKernelFamily::ReductionBuffer:
            return GfxKernelStageFamily::Reduction;
        case GfxMslKernelFamily::Conv2DDirectOrIm2col:
            return GfxKernelStageFamily::Convolution;
        case GfxMslKernelFamily::Unknown:
        default:
            return GfxKernelStageFamily::Unknown;
    }
}

}  // namespace

const char* gfx_msl_kernel_family_name(GfxMslKernelFamily family) {
    switch (family) {
        case GfxMslKernelFamily::EltwiseFusedBuffer:
            return "eltwise_fused_buffer";
        case GfxMslKernelFamily::TransposePackND:
            return "transpose_pack_nd";
        case GfxMslKernelFamily::ConcatSplitGeneric:
            return "concat_split_generic";
        case GfxMslKernelFamily::GatherScatterIndexed:
            return "gather_scatter_indexed";
        case GfxMslKernelFamily::RmsnormRopeFused:
            return "rmsnorm_rope_fused";
        case GfxMslKernelFamily::MaskedSoftmaxAttention:
            return "masked_softmax_attention";
        case GfxMslKernelFamily::KvCacheUpdate:
            return "kv_cache_update";
        case GfxMslKernelFamily::Conv3DDirectOrIm2col:
            return "conv3d_direct_or_im2col";
        case GfxMslKernelFamily::ReductionBuffer:
            return "reduction_buffer";
        case GfxMslKernelFamily::Conv2DDirectOrIm2col:
            return "conv2d_direct_or_im2col";
        case GfxMslKernelFamily::Unknown:
        default:
            return "unknown";
    }
}

const char* gfx_msl_required_kernel_entry_point(GfxMslKernelFamily family) {
    return gfx_msl_kernel_family_name(family);
}

uint32_t gfx_msl_kernel_family_abi_id(GfxMslKernelFamily family) {
    return static_cast<uint32_t>(family);
}

GfxMslExternalBufferAbiSpec gfx_msl_external_buffer_abi_spec(GfxMslKernelFamily family) {
    GfxMslExternalBufferAbiSpec spec{};
    switch (family) {
        case GfxMslKernelFamily::EltwiseFusedBuffer:
            return make_gfx_kernel_tail_outputs_abi();
        case GfxMslKernelFamily::MaskedSoftmaxAttention:
            return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                              GfxKernelBufferRole::TensorOutput,
                                              GfxKernelBufferRole::RuntimeParams});
        case GfxMslKernelFamily::ReductionBuffer:
            return make_gfx_kernel_leading_io_params_abi(/*input_count=*/1, /*output_count=*/1);
        case GfxMslKernelFamily::Conv2DDirectOrIm2col:
            return make_gfx_kernel_tail_outputs_abi();
        default:
            return spec;
    }
}

GfxMslExternalBufferAbiSpec gfx_msl_external_buffer_abi_spec(std::string_view stage_type,
                                                             std::string_view entry_point,
                                                             GfxMslKernelFamily family) {
    if (family == GfxMslKernelFamily::GatherScatterIndexed &&
        (stage_type == "TopK" || entry_point == "topk_kernel")) {
        return make_gfx_kernel_leading_io_params_abi(/*input_count=*/1, /*output_count=*/2);
    }
    if (family == GfxMslKernelFamily::GatherScatterIndexed &&
        (stage_type == "Gather" ||
         stage_type == "GatherND" ||
         stage_type == "GatherElements" ||
         entry_point == "gather_kernel" ||
         entry_point == "gathernd_kernel" ||
         entry_point == "gather_elements_kernel")) {
        return make_gfx_kernel_leading_io_params_abi(/*input_count=*/2, /*output_count=*/1);
    }
    if ((family == GfxMslKernelFamily::GatherScatterIndexed &&
         (stage_type == "Slice" ||
          stage_type == "StridedSlice" ||
          stage_type == "Tile" ||
          entry_point == "slice_kernel" ||
          entry_point == "tile_kernel")) ||
        (family == GfxMslKernelFamily::TransposePackND &&
         (stage_type == "Transpose" || entry_point == "transpose_kernel"))) {
        return make_gfx_kernel_leading_io_params_abi(/*input_count=*/1, /*output_count=*/1);
    }
    if (family == GfxMslKernelFamily::ConcatSplitGeneric) {
        if (entry_point == "concat_binary_kernel") {
            return make_gfx_kernel_leading_io_params_abi(/*input_count=*/2, /*output_count=*/1);
        }
        if (entry_point == "concat_kernel" || stage_type == "Split" || stage_type == "VariadicSplit") {
            return make_gfx_kernel_leading_io_params_abi(/*input_count=*/1, /*output_count=*/1);
        }
    }
    return gfx_msl_external_buffer_abi_spec(family);
}

GfxMslKernelFamily classify_msl_kernel_family(std::string_view stage_type,
                                              std::string_view entry_point) {
    if (stage_type == "Add" ||
        stage_type == "Subtract" ||
        stage_type == "Multiply" ||
        stage_type == "Divide" ||
        stage_type == "Power" ||
        stage_type == "Maximum" ||
        stage_type == "Minimum" ||
        stage_type == "Mod" ||
        stage_type == "FloorMod" ||
        stage_type == "PRelu" ||
        stage_type == "SquaredDifference" ||
        stage_type == "Equal" ||
        stage_type == "NotEqual" ||
        stage_type == "Greater" ||
        stage_type == "GreaterEqual" ||
        stage_type == "Less" ||
        stage_type == "LessEqual" ||
        stage_type == "LogicalAnd" ||
        stage_type == "LogicalOr" ||
        stage_type == "LogicalXor" ||
        stage_type == "Select" ||
        stage_type == "Broadcast" ||
        stage_type == "Relu" ||
        stage_type == "Sigmoid" ||
        stage_type == "Tanh" ||
        stage_type == "Elu" ||
        stage_type == "Gelu" ||
        stage_type == "Swish" ||
        stage_type == "HSwish" ||
        stage_type == "HSigmoid" ||
        stage_type == "SoftPlus" ||
        stage_type == "Mish" ||
        stage_type == "SoftSign" ||
        stage_type == "Abs" ||
        stage_type == "Sign" ||
        stage_type == "Clamp" ||
        stage_type == "LogicalNot" ||
        stage_type == "Exp" ||
        stage_type == "Log" ||
        stage_type == "Sqrt" ||
        stage_type == "Floor" ||
        stage_type == "Ceiling" ||
        stage_type == "Negative" ||
        stage_type == "Sin" ||
        stage_type == "Cos" ||
        stage_type == "Tan" ||
        stage_type == "Erf" ||
        stage_type == "Asin" ||
        stage_type == "Acos" ||
        stage_type == "Atan" ||
        stage_type == "Asinh" ||
        stage_type == "Acosh" ||
        stage_type == "Atanh" ||
        stage_type == "Sinh" ||
        stage_type == "Cosh" ||
        stage_type == "Round" ||
        stage_type == "Convert" ||
        entry_point == "eltwise_kernel" ||
        entry_point == "unary_kernel" ||
        entry_point == "select_kernel" ||
        entry_point == "broadcast_kernel" ||
        entry_point == "convert_kernel") {
        return GfxMslKernelFamily::EltwiseFusedBuffer;
    }
    if (stage_type == "Transpose" ||
        stage_type == "Reshape" ||
        stage_type == "Interpolate" ||
        stage_type == "DepthToSpace" ||
        stage_type == "SpaceToDepth" ||
        entry_point == "transpose_kernel" ||
        entry_point == "interpolate_kernel" ||
        entry_point == "depth_to_space_kernel" ||
        entry_point == "space_to_depth_kernel") {
        return GfxMslKernelFamily::TransposePackND;
    }
    if (stage_type == "Concat" ||
        stage_type == "Split" ||
        stage_type == "VariadicSplit" ||
        entry_point == "concat_kernel" ||
        entry_point == "concat_binary_kernel" ||
        entry_point == "split_kernel") {
        return GfxMslKernelFamily::ConcatSplitGeneric;
    }
    if (stage_type == "Gather" ||
        stage_type == "GatherND" ||
        stage_type == "GatherElements" ||
        stage_type == "ScatterElementsUpdate" ||
        stage_type == "ScatterUpdate" ||
        stage_type == "ScatterNDUpdate" ||
        stage_type == "Slice" ||
        stage_type == "StridedSlice" ||
        stage_type == "Tile" ||
        stage_type == "Range" ||
        stage_type == "ShapeOf" ||
        stage_type == "TopK" ||
        stage_type == "Pad" ||
        stage_type == "Reverse" ||
        entry_point == "gather_kernel" ||
        entry_point == "gathernd_kernel" ||
        entry_point == "gather_elements_kernel" ||
        entry_point == "scatter_update_kernel" ||
        entry_point == "scatter_elements_update" ||
        entry_point == "scatter_nd_update" ||
        entry_point == "slice_kernel" ||
        entry_point == "tile_kernel" ||
        entry_point == "range_kernel" ||
        entry_point == "shapeof_kernel" ||
        entry_point == "topk_kernel" ||
        entry_point == "pad_kernel" ||
        entry_point == "reverse_kernel") {
        return GfxMslKernelFamily::GatherScatterIndexed;
    }
    if (stage_type == "RMS" ||
        stage_type == "RMSNorm" ||
        stage_type == "RoPE" ||
        stage_type == "RotaryEmbedding" ||
        entry_point == "rms_kernel" ||
        entry_point == "rope_kernel") {
        return GfxMslKernelFamily::RmsnormRopeFused;
    }
    if (stage_type == "Softmax" ||
        stage_type == "LogSoftmax" ||
        stage_type == "ScaledDotProductAttention" ||
        entry_point == "softmax_kernel" ||
        entry_point == "sdpa_kernel" ||
        entry_point == "sdpa_causal_mask_kernel") {
        return GfxMslKernelFamily::MaskedSoftmaxAttention;
    }
    if (stage_type == "KVCache" ||
        stage_type == "ReadValue" ||
        stage_type == "Assign" ||
        entry_point == "kv_cache_update") {
        return GfxMslKernelFamily::KvCacheUpdate;
    }
    if (stage_type == "Convolution3D" ||
        stage_type == "Conv3D" ||
        entry_point == "conv3d_kernel") {
        return GfxMslKernelFamily::Conv3DDirectOrIm2col;
    }
    if (stage_type == "Convolution" ||
        entry_point == "conv2d_kernel") {
        return GfxMslKernelFamily::Conv2DDirectOrIm2col;
    }
    if (stage_type == "ReduceSum" ||
        stage_type == "ReduceMean" ||
        stage_type == "ReduceMax" ||
        stage_type == "ReduceMin" ||
        stage_type == "ReduceProd" ||
        stage_type == "ReduceL1" ||
        stage_type == "ReduceL2" ||
        entry_point == "reduce_kernel") {
        return GfxMslKernelFamily::ReductionBuffer;
    }
    return GfxMslKernelFamily::Unknown;
}

GfxMslKernelPlan make_msl_kernel_plan(std::string_view stage_type,
                                      std::string_view entry_point) {
    GfxMslKernelPlan plan{};
    plan.family = classify_msl_kernel_family(stage_type, entry_point);
    plan.valid = plan.family != GfxMslKernelFamily::Unknown;
    plan.family_name = gfx_msl_kernel_family_name(plan.family);
    plan.required_entry_point = gfx_msl_required_kernel_entry_point(plan.family);
    plan.abi_kernel_family = gfx_msl_kernel_family_abi_id(plan.family);
    plan.external_buffer_abi = gfx_msl_external_buffer_abi_spec(stage_type, entry_point, plan.family);
    plan.dispatch_flags = plan.precompiled_metallib_required
                              ? GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired
                              : GfxMpsrtMslDispatchFlagNone;
    if (plan.family == GfxMslKernelFamily::MaskedSoftmaxAttention) {
        plan.threads_per_threadgroup = 128;
    } else if (plan.family == GfxMslKernelFamily::RmsnormRopeFused) {
        plan.threads_per_threadgroup = 256;
    } else if (plan.family == GfxMslKernelFamily::Conv3DDirectOrIm2col) {
        plan.threads_per_threadgroup = 128;
    } else if (plan.family == GfxMslKernelFamily::ReductionBuffer) {
        plan.threads_per_threadgroup = 128;
    }
    if (plan.valid) {
        plan.kernel_manifest = make_gfx_custom_kernel_manifest(plan.family_name,
                                                              plan.abi_kernel_family,
                                                              plan.required_entry_point,
                                                              plan.external_buffer_abi,
                                                              plan.threads_per_threadgroup,
                                                              plan.precompiled_metallib_required);
        std::string specialization_key = "apple_msl:buffer:";
        specialization_key += stage_type;
        plan.stage_manifest = make_gfx_custom_kernel_stage_manifest(
            gfx_msl_stage_family_from_kernel_family(plan.family),
            GfxKernelBackendDomain::AppleMsl,
            GfxKernelStorageKind::Buffer,
            std::move(specialization_key),
            plan.kernel_manifest);
    }
    return plan;
}

}  // namespace gfx_plugin
}  // namespace ov
