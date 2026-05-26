// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/gfx_custom_kernel_families.hpp"

#include <string>
#include <utility>

namespace ov {
namespace gfx_plugin {
namespace {

bool is_gfx_kernel_binary_eltwise_stage(std::string_view stage_type) {
  return stage_type == "Add" || stage_type == "Subtract" ||
         stage_type == "Multiply" || stage_type == "Divide" ||
         stage_type == "Power" || stage_type == "Maximum" ||
         stage_type == "Minimum" || stage_type == "Mod" ||
         stage_type == "FloorMod" || stage_type == "PRelu" ||
         stage_type == "SquaredDifference" || stage_type == "Equal" ||
         stage_type == "NotEqual" || stage_type == "Greater" ||
         stage_type == "GreaterEqual" || stage_type == "Less" ||
         stage_type == "LessEqual" || stage_type == "LogicalAnd" ||
         stage_type == "LogicalOr" || stage_type == "LogicalXor";
}

bool is_gfx_kernel_unary_eltwise_stage(std::string_view stage_type) {
  return stage_type == "Relu" || stage_type == "Sigmoid" ||
         stage_type == "Tanh" || stage_type == "Elu" || stage_type == "Gelu" ||
         stage_type == "Swish" || stage_type == "HSwish" ||
         stage_type == "HSigmoid" || stage_type == "SoftPlus" ||
         stage_type == "Mish" || stage_type == "SoftSign" ||
         stage_type == "Abs" || stage_type == "Sign" || stage_type == "Clamp" ||
         stage_type == "LogicalNot" || stage_type == "Exp" ||
         stage_type == "Log" || stage_type == "Sqrt" || stage_type == "Floor" ||
         stage_type == "Ceiling" || stage_type == "Negative" ||
         stage_type == "Sin" || stage_type == "Cos" || stage_type == "Tan" ||
         stage_type == "Erf" || stage_type == "Asin" || stage_type == "Acos" ||
         stage_type == "Atan" || stage_type == "Asinh" ||
         stage_type == "Acosh" || stage_type == "Atanh" ||
         stage_type == "Sinh" || stage_type == "Cosh" ||
         stage_type == "Round" || stage_type == "Convert";
}

GfxKernelExternalBufferAbiSpec make_gfx_kernel_unary_eltwise_abi() {
  return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                    GfxKernelBufferRole::TensorOutput,
                                    GfxKernelBufferRole::ScalarParam});
}

GfxKernelExternalBufferAbiSpec make_gfx_kernel_binary_eltwise_abi() {
  return make_gfx_kernel_roles_abi(
      {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
       GfxKernelBufferRole::TensorOutput, GfxKernelBufferRole::ScalarParam,
       GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::RuntimeParams,
       GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams});
}

} // namespace

const char *gfx_kernel_family_name(GfxKernelFamily family) {
  switch (family) {
  case GfxKernelFamily::EltwiseFusedBuffer:
    return "eltwise_fused_buffer";
  case GfxKernelFamily::TransposePackND:
    return "transpose_pack_nd";
  case GfxKernelFamily::ConcatSplitGeneric:
    return "concat_split_generic";
  case GfxKernelFamily::GatherScatterIndexed:
    return "gather_scatter_indexed";
  case GfxKernelFamily::RmsnormRopeFused:
    return "rmsnorm_rope_fused";
  case GfxKernelFamily::MaskedSoftmaxAttention:
    return "masked_softmax_attention";
  case GfxKernelFamily::KvCacheUpdate:
    return "kv_cache_update";
  case GfxKernelFamily::Conv3DDirect:
    return "conv3d_direct";
  case GfxKernelFamily::ReductionBuffer:
    return "reduction_buffer";
  case GfxKernelFamily::Conv2DDirect:
    return "conv2d_direct";
  case GfxKernelFamily::MatMulBuffer:
    return "matmul_buffer";
  case GfxKernelFamily::Pool2DWindow:
    return "pool2d_window";
  case GfxKernelFamily::BatchNormBuffer:
    return "batchnorm_buffer";
  case GfxKernelFamily::Unknown:
  default:
    return "unknown";
  }
}

const char *gfx_kernel_required_entry_point(GfxKernelFamily family) {
  return gfx_kernel_family_name(family);
}

uint32_t gfx_kernel_family_abi_id(GfxKernelFamily family) {
  return static_cast<uint32_t>(family);
}

GfxKernelStageFamily
gfx_kernel_stage_family_from_kernel_family(GfxKernelFamily family) {
  switch (family) {
  case GfxKernelFamily::EltwiseFusedBuffer:
    return GfxKernelStageFamily::Eltwise;
  case GfxKernelFamily::TransposePackND:
    return GfxKernelStageFamily::Transpose;
  case GfxKernelFamily::ConcatSplitGeneric:
    return GfxKernelStageFamily::ConcatSplit;
  case GfxKernelFamily::GatherScatterIndexed:
    return GfxKernelStageFamily::GatherScatter;
  case GfxKernelFamily::RmsnormRopeFused:
    return GfxKernelStageFamily::RmsnormRope;
  case GfxKernelFamily::MaskedSoftmaxAttention:
    return GfxKernelStageFamily::AttentionSoftmax;
  case GfxKernelFamily::KvCacheUpdate:
    return GfxKernelStageFamily::KvCache;
  case GfxKernelFamily::Conv3DDirect:
    return GfxKernelStageFamily::Conv3D;
  case GfxKernelFamily::ReductionBuffer:
    return GfxKernelStageFamily::Reduction;
  case GfxKernelFamily::Conv2DDirect:
    return GfxKernelStageFamily::Convolution;
  case GfxKernelFamily::MatMulBuffer:
    return GfxKernelStageFamily::Gemm;
  case GfxKernelFamily::Pool2DWindow:
    return GfxKernelStageFamily::Pooling;
  case GfxKernelFamily::BatchNormBuffer:
    return GfxKernelStageFamily::Eltwise;
  case GfxKernelFamily::Unknown:
  default:
    return GfxKernelStageFamily::Unknown;
  }
}

GfxKernelExternalBufferAbiSpec
gfx_kernel_external_buffer_abi_spec_for_family(GfxKernelFamily family) {
  GfxKernelExternalBufferAbiSpec spec{};
  switch (family) {
  case GfxKernelFamily::EltwiseFusedBuffer:
    return spec;
  case GfxKernelFamily::MaskedSoftmaxAttention:
    return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorOutput,
                                      GfxKernelBufferRole::RuntimeParams});
  case GfxKernelFamily::ReductionBuffer:
    return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorOutput,
                                      GfxKernelBufferRole::RuntimeParams});
  case GfxKernelFamily::Conv2DDirect:
    return spec;
  case GfxKernelFamily::MatMulBuffer:
    return spec;
  case GfxKernelFamily::Pool2DWindow:
    return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::RuntimeParams,
                                      GfxKernelBufferRole::TensorOutput});
  case GfxKernelFamily::BatchNormBuffer:
    return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::ConstTensor,
                                      GfxKernelBufferRole::TensorOutput,
                                      GfxKernelBufferRole::RuntimeParams});
  default:
    return spec;
  }
}

GfxKernelExternalBufferAbiSpec
gfx_kernel_external_buffer_abi_spec_for_stage(std::string_view stage_type,
                                              std::string_view entry_point,
                                              GfxKernelFamily family) {
  if (family == GfxKernelFamily::Conv2DDirect &&
      (stage_type == "Convolution" || stage_type == "GroupConvolution" ||
       stage_type == "GroupConv2D" || entry_point == "conv2d_kernel")) {
    return make_gfx_kernel_roles_abi(
        {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::ConstTensor,
         GfxKernelBufferRole::ConstTensor, GfxKernelBufferRole::ConstTensor,
         GfxKernelBufferRole::ConstTensor, GfxKernelBufferRole::ConstTensor,
         GfxKernelBufferRole::ConstTensor, GfxKernelBufferRole::RuntimeParams,
         GfxKernelBufferRole::TensorOutput});
  }
  if (family == GfxKernelFamily::Conv3DDirect &&
      entry_point == "conv3d_kernel") {
    return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::ConstTensor,
                                      GfxKernelBufferRole::TensorOutput,
                                      GfxKernelBufferRole::RuntimeParams});
  }
  if (family == GfxKernelFamily::EltwiseFusedBuffer) {
    if (entry_point == "unary_kernel" || entry_point == "convert_kernel" ||
        is_gfx_kernel_unary_eltwise_stage(stage_type)) {
      return make_gfx_kernel_unary_eltwise_abi();
    }
    if (entry_point == "broadcast_kernel") {
      return make_gfx_kernel_roles_abi(
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
           GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::ScalarParam,
           GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams});
    }
    if (entry_point == "select_kernel") {
      return make_gfx_kernel_roles_abi(
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
           GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
           GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::ScalarParam,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams});
    }
    if (entry_point == "eltwise_kernel" ||
        is_gfx_kernel_binary_eltwise_stage(stage_type)) {
      return make_gfx_kernel_binary_eltwise_abi();
    }
  }
  if (family == GfxKernelFamily::MatMulBuffer &&
      entry_point == "compressed_matmul_kernel") {
    return make_gfx_kernel_roles_abi(
        {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::ConstTensor,
         GfxKernelBufferRole::ConstTensor, GfxKernelBufferRole::TensorOutput});
  }
  if (family == GfxKernelFamily::MatMulBuffer &&
      (stage_type == "MatMul" || entry_point == "matmul_kernel")) {
    return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorOutput});
  }
  if (family == GfxKernelFamily::RmsnormRopeFused &&
      entry_point == "rms_kernel") {
    if (stage_type == "RMSResidual") {
      return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput});
    }
    return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorOutput});
  }
  if (family == GfxKernelFamily::RmsnormRopeFused &&
      entry_point == "rope_kernel") {
    if (stage_type == "RoPEWithPosition") {
      return make_gfx_kernel_roles_abi(
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
           GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
           GfxKernelBufferRole::TensorOutput});
    }
    return make_gfx_kernel_roles_abi(
        {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
         GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput});
  }
  if (family == GfxKernelFamily::ReductionBuffer &&
      entry_point == "reduce_kernel") {
    return make_gfx_kernel_roles_abi(
        {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
         GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::ScalarParam,
         GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
         GfxKernelBufferRole::RuntimeParams, GfxKernelBufferRole::RuntimeParams,
         GfxKernelBufferRole::RuntimeParams});
  }
  if (family == GfxKernelFamily::GatherScatterIndexed &&
      (entry_point == "scatter_elements_init" ||
       entry_point == "scatter_nd_init")) {
    return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorOutput,
                                      GfxKernelBufferRole::RuntimeParams});
  }
  if (family == GfxKernelFamily::GatherScatterIndexed &&
      (stage_type == "ScatterElementsUpdate" ||
       stage_type == "ScatterNDUpdate" ||
       entry_point == "scatter_elements_update" ||
       entry_point == "scatter_nd_update")) {
    return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorOutput,
                                      GfxKernelBufferRole::RuntimeParams});
  }
  if (family == GfxKernelFamily::GatherScatterIndexed &&
      (stage_type == "ScatterUpdate" ||
       entry_point == "scatter_update_kernel")) {
    return make_gfx_kernel_roles_abi(
        {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
         GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
         GfxKernelBufferRole::RuntimeParams});
  }
  if (family == GfxKernelFamily::GatherScatterIndexed &&
      (stage_type == "TopK" || entry_point == "topk_kernel")) {
    return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorOutput,
                                      GfxKernelBufferRole::TensorOutput});
  }
  if (family == GfxKernelFamily::GatherScatterIndexed &&
      (stage_type == "Gather" || stage_type == "GatherND" ||
       stage_type == "GatherElements" || entry_point == "gather_kernel" ||
       entry_point == "gathernd_kernel" ||
       entry_point == "gather_elements_kernel")) {
    return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorOutput,
                                      GfxKernelBufferRole::RuntimeParams});
  }
  if ((family == GfxKernelFamily::GatherScatterIndexed &&
       (stage_type == "Slice" || stage_type == "StridedSlice" ||
        stage_type == "Tile" || stage_type == "Pad" ||
        stage_type == "Reverse" || stage_type == "ShapeOf" ||
        stage_type == "Range" || entry_point == "slice_kernel" ||
        entry_point == "tile_kernel" || entry_point == "pad_kernel" ||
        entry_point == "reverse_kernel" || entry_point == "shapeof_kernel" ||
        entry_point == "range_kernel")) ||
      (family == GfxKernelFamily::TransposePackND &&
       (stage_type == "Transpose" || stage_type == "Reshape" ||
        stage_type == "DepthToSpace" || stage_type == "SpaceToDepth" ||
        entry_point == "transpose_kernel" ||
        entry_point == "depth_to_space_kernel" ||
        entry_point == "space_to_depth_kernel"))) {
    if (stage_type == "Slice" || stage_type == "StridedSlice" ||
        entry_point == "slice_kernel") {
      return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams,
                                        GfxKernelBufferRole::RuntimeParams,
                                        GfxKernelBufferRole::RuntimeParams,
                                        GfxKernelBufferRole::RuntimeParams,
                                        GfxKernelBufferRole::RuntimeParams,
                                        GfxKernelBufferRole::RuntimeParams});
    }
    if (stage_type == "Tile" || entry_point == "tile_kernel") {
      return make_gfx_kernel_roles_abi(
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
           GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::ScalarParam,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams});
    }
    if (stage_type == "Pad" || entry_point == "pad_kernel") {
      return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams,
                                        GfxKernelBufferRole::RuntimeParams,
                                        GfxKernelBufferRole::RuntimeParams,
                                        GfxKernelBufferRole::RuntimeParams,
                                        GfxKernelBufferRole::RuntimeParams,
                                        GfxKernelBufferRole::RuntimeParams,
                                        GfxKernelBufferRole::RuntimeParams,
                                        GfxKernelBufferRole::RuntimeParams});
    }
    if (stage_type == "ShapeOf" || entry_point == "shapeof_kernel") {
      return make_gfx_kernel_roles_abi(
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
           GfxKernelBufferRole::ScalarParam,
           GfxKernelBufferRole::RuntimeParams});
    }
    if (stage_type == "Range" || entry_point == "range_kernel") {
      return make_gfx_kernel_roles_abi(
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
           GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
           GfxKernelBufferRole::ScalarParam});
    }
    return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorOutput,
                                      GfxKernelBufferRole::RuntimeParams});
  }
  if (family == GfxKernelFamily::TransposePackND &&
      (stage_type == "Interpolate" || entry_point == "interpolate_kernel")) {
    return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorOutput,
                                      GfxKernelBufferRole::RuntimeParams});
  }
  if (family == GfxKernelFamily::ConcatSplitGeneric) {
    if (entry_point == "concat_binary_kernel") {
      return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams});
    }
    if (entry_point == "concat_kernel" || stage_type == "Split" ||
        stage_type == "VariadicSplit") {
      return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams});
    }
  }
  if (family == GfxKernelFamily::MaskedSoftmaxAttention &&
      entry_point == "sdpa_kernel") {
    return make_gfx_kernel_roles_abi(
        {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
         GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
         GfxKernelBufferRole::RuntimeParams,
         GfxKernelBufferRole::TensorOutput});
  }
  if (family == GfxKernelFamily::MaskedSoftmaxAttention &&
      entry_point == "sdpa_nomask_kernel") {
    return make_gfx_kernel_roles_abi(
        {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
         GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::RuntimeParams,
         GfxKernelBufferRole::TensorOutput});
  }
  if (family == GfxKernelFamily::MaskedSoftmaxAttention &&
      entry_point == "sdpa_causal_mask_kernel") {
    return make_gfx_kernel_roles_abi(
        {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
         GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
         GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::RuntimeParams,
         GfxKernelBufferRole::TensorOutput});
  }
  return gfx_kernel_external_buffer_abi_spec_for_family(family);
}

GfxKernelDispatchPolicy
gfx_kernel_dispatch_policy_for_family(GfxKernelFamily family) {
  uint32_t threads_per_threadgroup = 256;
  GfxKernelDispatchGrid grid = GfxKernelDispatchGrid::Linear1D;
  switch (family) {
  case GfxKernelFamily::MaskedSoftmaxAttention:
  case GfxKernelFamily::Conv3DDirect:
  case GfxKernelFamily::ReductionBuffer:
    threads_per_threadgroup = 128;
    break;
  case GfxKernelFamily::Pool2DWindow:
    grid = GfxKernelDispatchGrid::Tiled2D;
    threads_per_threadgroup = 256;
    break;
  case GfxKernelFamily::TransposePackND:
    grid = GfxKernelDispatchGrid::Tiled2D;
    threads_per_threadgroup = 256;
    break;
  case GfxKernelFamily::Conv2DDirect:
    grid = GfxKernelDispatchGrid::Tiled2D;
    threads_per_threadgroup = 256;
    break;
  case GfxKernelFamily::Unknown:
    return {};
  default:
    break;
  }
  return make_gfx_kernel_dispatch_policy(grid, threads_per_threadgroup,
                                         /*precompiled_binary_required=*/true);
}

GfxKernelDispatchPolicy
gfx_kernel_dispatch_policy_for_stage(std::string_view /*stage_type*/,
                                     std::string_view /*entry_point*/,
                                     GfxKernelFamily family) {
  return gfx_kernel_dispatch_policy_for_family(family);
}

GfxKernelFamily
classify_gfx_custom_kernel_family(std::string_view stage_type,
                                  std::string_view entry_point) {
  if (stage_type == "Add" || stage_type == "Subtract" ||
      stage_type == "Multiply" || stage_type == "Divide" ||
      stage_type == "Power" || stage_type == "Maximum" ||
      stage_type == "Minimum" || stage_type == "Mod" ||
      stage_type == "FloorMod" || stage_type == "PRelu" ||
      stage_type == "SquaredDifference" || stage_type == "Equal" ||
      stage_type == "NotEqual" || stage_type == "Greater" ||
      stage_type == "GreaterEqual" || stage_type == "Less" ||
      stage_type == "LessEqual" || stage_type == "LogicalAnd" ||
      stage_type == "LogicalOr" || stage_type == "LogicalXor" ||
      stage_type == "Select" || stage_type == "Broadcast" ||
      stage_type == "Relu" || stage_type == "Sigmoid" || stage_type == "Tanh" ||
      stage_type == "Elu" || stage_type == "Gelu" || stage_type == "Swish" ||
      stage_type == "HSwish" || stage_type == "HSigmoid" ||
      stage_type == "SoftPlus" || stage_type == "Mish" ||
      stage_type == "SoftSign" || stage_type == "Abs" || stage_type == "Sign" ||
      stage_type == "Clamp" || stage_type == "LogicalNot" ||
      stage_type == "Exp" || stage_type == "Log" || stage_type == "Sqrt" ||
      stage_type == "Floor" || stage_type == "Ceiling" ||
      stage_type == "Negative" || stage_type == "Sin" || stage_type == "Cos" ||
      stage_type == "Tan" || stage_type == "Erf" || stage_type == "Asin" ||
      stage_type == "Acos" || stage_type == "Atan" || stage_type == "Asinh" ||
      stage_type == "Acosh" || stage_type == "Atanh" || stage_type == "Sinh" ||
      stage_type == "Cosh" || stage_type == "Round" ||
      stage_type == "Convert" || entry_point == "eltwise_kernel" ||
      entry_point == "eltwise_fused_buffer" ||
      entry_point == "unary_kernel" || entry_point == "select_kernel" ||
      entry_point == "broadcast_kernel" || entry_point == "convert_kernel" ||
      stage_type == "ConvTextureSwishEpilogue" ||
      entry_point == "gfx_mpsrt_conv_texture_swish_epilogue") {
    return GfxKernelFamily::EltwiseFusedBuffer;
  }
  if (stage_type == "Transpose" || stage_type == "Reshape" ||
      stage_type == "Interpolate" || stage_type == "DepthToSpace" ||
      stage_type == "SpaceToDepth" || entry_point == "transpose_kernel" ||
      entry_point == "interpolate_kernel" ||
      entry_point == "depth_to_space_kernel" ||
      entry_point == "space_to_depth_kernel") {
    return GfxKernelFamily::TransposePackND;
  }
  if (stage_type == "Concat" || stage_type == "Split" ||
      stage_type == "VariadicSplit" || entry_point == "concat_kernel" ||
      entry_point == "concat_binary_kernel" || entry_point == "split_kernel") {
    return GfxKernelFamily::ConcatSplitGeneric;
  }
  if (stage_type == "Gather" || stage_type == "GatherND" ||
      stage_type == "GatherElements" || stage_type == "ScatterElementsUpdate" ||
      stage_type == "ScatterUpdate" || stage_type == "ScatterNDUpdate" ||
      stage_type == "Slice" || stage_type == "StridedSlice" ||
      stage_type == "Tile" || stage_type == "Range" ||
      stage_type == "ShapeOf" || stage_type == "TopK" || stage_type == "Pad" ||
      stage_type == "Reverse" || entry_point == "gather_kernel" ||
      entry_point == "gathernd_kernel" ||
      entry_point == "gather_elements_kernel" ||
      entry_point == "scatter_update_kernel" ||
      entry_point == "scatter_elements_init" ||
      entry_point == "scatter_elements_update" ||
      entry_point == "scatter_nd_init" || entry_point == "scatter_nd_update" ||
      entry_point == "slice_kernel" || entry_point == "tile_kernel" ||
      entry_point == "range_kernel" || entry_point == "shapeof_kernel" ||
      entry_point == "topk_kernel" || entry_point == "pad_kernel" ||
      entry_point == "reverse_kernel") {
    return GfxKernelFamily::GatherScatterIndexed;
  }
  if (stage_type == "RMS" || stage_type == "RMSNorm" ||
      stage_type == "RMSResidual" || stage_type == "RoPE" ||
      stage_type == "RoPEWithPosition" || stage_type == "RotaryEmbedding" ||
      entry_point == "rms_kernel" || entry_point == "rope_kernel") {
    return GfxKernelFamily::RmsnormRopeFused;
  }
  if (stage_type == "Softmax" || stage_type == "LogSoftmax" ||
      stage_type == "ScaledDotProductAttention" ||
      entry_point == "softmax_kernel" || entry_point == "sdpa_kernel" ||
      entry_point == "sdpa_nomask_kernel" ||
      entry_point == "sdpa_causal_mask_kernel") {
    return GfxKernelFamily::MaskedSoftmaxAttention;
  }
  if (stage_type == "KVCache" || stage_type == "ReadValue" ||
      stage_type == "Assign" || entry_point == "kv_cache_update") {
    return GfxKernelFamily::KvCacheUpdate;
  }
  if (stage_type == "Convolution3D" || stage_type == "Conv3D" ||
      entry_point == "conv3d_kernel") {
    return GfxKernelFamily::Conv3DDirect;
  }
  if (stage_type == "Convolution" || stage_type == "GroupConvolution" ||
      stage_type == "GroupConv2D" || entry_point == "conv2d_kernel") {
    return GfxKernelFamily::Conv2DDirect;
  }
  if (stage_type == "MatMul" || entry_point == "matmul_kernel" ||
      entry_point == "compressed_matmul_kernel") {
    return GfxKernelFamily::MatMulBuffer;
  }
  if (stage_type == "MaxPool" || stage_type == "AvgPool" ||
      entry_point == "pool2d_kernel") {
    return GfxKernelFamily::Pool2DWindow;
  }
  if (stage_type == "BatchNormInference" ||
      entry_point == "batchnorm2d_kernel") {
    return GfxKernelFamily::BatchNormBuffer;
  }
  if (stage_type == "ReduceSum" || stage_type == "ReduceMean" ||
      stage_type == "ReduceMax" || stage_type == "ReduceMin" ||
      stage_type == "ReduceProd" || stage_type == "ReduceL1" ||
      stage_type == "ReduceL2" || stage_type == "ReduceLogicalAnd" ||
      stage_type == "ReduceLogicalOr" || entry_point == "reduce_kernel") {
    return GfxKernelFamily::ReductionBuffer;
  }
  return GfxKernelFamily::Unknown;
}

GfxCustomKernelStagePlan make_gfx_custom_kernel_stage_plan(
    std::string_view stage_type, std::string_view entry_point,
    GfxKernelBackendDomain backend_domain, GfxKernelStorageKind storage,
    std::string_view specialization_prefix) {
  GfxCustomKernelStagePlan plan{};
  plan.family = classify_gfx_custom_kernel_family(stage_type, entry_point);
  plan.valid = plan.family != GfxKernelFamily::Unknown;
  if (!plan.valid) {
    return plan;
  }

  const std::string family_name = gfx_kernel_family_name(plan.family);
  const std::string required_entry_point =
      gfx_kernel_required_entry_point(plan.family);
  const uint32_t abi_kernel_family = gfx_kernel_family_abi_id(plan.family);
  const auto external_buffer_abi =
      gfx_kernel_external_buffer_abi_spec_for_stage(stage_type, entry_point,
                                                    plan.family);
  const auto dispatch_policy = gfx_kernel_dispatch_policy_for_stage(
      stage_type, entry_point, plan.family);

  auto kernel_manifest = make_gfx_custom_kernel_manifest(
      family_name, abi_kernel_family, required_entry_point, external_buffer_abi,
      dispatch_policy);
  std::string specialization_key(specialization_prefix);
  specialization_key += stage_type;
  plan.stage_manifest = make_gfx_custom_kernel_stage_manifest(
      gfx_kernel_stage_family_from_kernel_family(plan.family), backend_domain,
      storage, std::move(specialization_key), std::move(kernel_manifest));
  return plan;
}

} // namespace gfx_plugin
} // namespace ov
