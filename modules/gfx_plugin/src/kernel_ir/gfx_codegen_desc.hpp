// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include <string>

#include "runtime/gfx_activation.hpp"

namespace ov {
namespace gfx_plugin {

struct BaseCodegenDesc {
  ov::element::Type element_type = ov::element::f16;
};

struct MatMulCodegenDesc : BaseCodegenDesc {
  ov::element::Type input_a_type = ov::element::dynamic;
  ov::element::Type input_b_type = ov::element::dynamic;
  ov::element::Type bias_type = ov::element::dynamic;
  ov::element::Type output_type = ov::element::dynamic;
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  int64_t batch = 1;
  int64_t batch_a = 1;
  int64_t batch_b = 1;
  bool a_transpose = false;
  bool b_transpose = false;
  bool b_is_nk_layout = false;
  bool has_bias = false;
  std::array<int64_t, 3> bias_dims{{1, 1, 1}}; // aligned to [batch, M, N]
  bool has_activation = false;
  ActivationKind activation = ActivationKind::Relu;
  float alpha = 0.0f;
};

inline bool gfx_matmul_float_like_type(const ov::element::Type &type) {
  return type == ov::element::dynamic || type == ov::element::f16 ||
         type == ov::element::f32;
}

inline uint32_t
gfx_matmul_parallel_reduction_threads(const MatMulCodegenDesc &desc) {
  if (desc.M != 1 || desc.N < 16 || desc.K < 512) {
    return 1;
  }
  if (!gfx_matmul_float_like_type(desc.input_a_type) ||
      !gfx_matmul_float_like_type(desc.input_b_type) ||
      !gfx_matmul_float_like_type(desc.output_type)) {
    return 1;
  }
  return desc.K >= 1024 ? 256u : 128u;
}

inline uint64_t gfx_matmul_dispatch_items(const MatMulCodegenDesc &desc) {
  const uint64_t outputs = static_cast<uint64_t>(desc.batch) *
                           static_cast<uint64_t>(desc.M) *
                           static_cast<uint64_t>(desc.N);
  return outputs * gfx_matmul_parallel_reduction_threads(desc);
}

struct Conv2DCodegenDesc : BaseCodegenDesc {
  ov::element::Type input_type = ov::element::dynamic;
  ov::element::Type weight_type = ov::element::dynamic;
  ov::element::Type bias_type = ov::element::dynamic;
  ov::element::Type bn_type = ov::element::dynamic;
  ov::element::Type output_type = ov::element::dynamic;
  uint32_t N = 0;
  uint32_t C_in = 0;
  uint32_t H = 0;
  uint32_t W = 0;
  uint32_t C_out = 0;
  // Group convolution parameters. For non-group conv they mirror C_in/C_out.
  uint32_t groups = 1;
  uint32_t C_in_pg = 0;
  uint32_t C_out_pg = 0;
  uint32_t kH = 0;
  uint32_t kW = 0;
  uint32_t strideH = 1;
  uint32_t strideW = 1;
  uint32_t dilationH = 1;
  uint32_t dilationW = 1;
  uint32_t padTop = 0;
  uint32_t padLeft = 0;
  uint32_t padBottom = 0;
  uint32_t padRight = 0;
  uint32_t outH = 0;
  uint32_t outW = 0;
  bool has_bias = false;
  uint32_t bias_rank = 1; // 1 or 4
  bool has_activation = false;
  ActivationKind activation = ActivationKind::Relu;
  float alpha = 0.0f;
  uint32_t output_channels_per_thread = 1;
  uint32_t output_width_per_thread = 1;
  bool use_special_k3 = false; // enable k=3 stride1/2 optimized kernel
  // BatchNorm + clamp support for fused conv
  bool has_bn = false;
  float epsilon = 0.0f;
  float clamp_min = 0.0f;
  float clamp_max = 0.0f;
  std::vector<float> gamma;
  std::vector<float> beta;
  std::vector<float> mean;
  std::vector<float> var;
};

inline bool gfx_conv2d_float_like_type(const ov::element::Type &type) {
  return type == ov::element::dynamic || type == ov::element::f16 ||
         type == ov::element::f32;
}

inline uint32_t gfx_conv2d_output_channel_block(const Conv2DCodegenDesc &desc) {
  if (desc.groups > 1 || desc.C_out < 4) {
    return 1;
  }
  if (!gfx_conv2d_float_like_type(desc.input_type) ||
      !gfx_conv2d_float_like_type(desc.weight_type) ||
      !gfx_conv2d_float_like_type(desc.output_type)) {
    return 1;
  }
  return 4;
}

inline uint32_t gfx_conv2d_output_width_block(const Conv2DCodegenDesc &desc) {
  if (gfx_conv2d_output_channel_block(desc) <= 1 || desc.outW < 2) {
    return 1;
  }
  return 2;
}

inline uint64_t gfx_conv2d_dispatch_items(uint64_t n, uint64_t c_out,
                                          uint64_t out_h, uint64_t out_w,
                                          uint32_t output_channels_per_thread,
                                          uint32_t output_width_per_thread) {
  const uint64_t channel_block =
      std::max<uint32_t>(output_channels_per_thread, 1u);
  const uint64_t width_block = std::max<uint32_t>(output_width_per_thread, 1u);
  const uint64_t width_blocks = (out_w + width_block - 1) / width_block;
  const uint64_t spatial = n * out_h * width_blocks;
  const uint64_t channel_blocks = (c_out + channel_block - 1) / channel_block;
  return spatial * channel_blocks;
}

inline uint64_t gfx_conv2d_dispatch_items(const Conv2DCodegenDesc &desc) {
  const uint32_t channel_block = desc.output_channels_per_thread
                                     ? desc.output_channels_per_thread
                                     : gfx_conv2d_output_channel_block(desc);
  const uint32_t width_block = desc.output_width_per_thread
                                   ? desc.output_width_per_thread
                                   : gfx_conv2d_output_width_block(desc);
  return gfx_conv2d_dispatch_items(desc.N, desc.C_out, desc.outH, desc.outW,
                                   channel_block, width_block);
}

struct Conv3DCodegenDesc : BaseCodegenDesc {
  uint32_t N = 0;
  uint32_t C_in = 0;
  uint32_t D = 0;
  uint32_t H = 0;
  uint32_t W = 0;
  uint32_t C_out = 0;
  uint32_t kD = 0;
  uint32_t kH = 0;
  uint32_t kW = 0;
  uint32_t strideD = 1;
  uint32_t strideH = 1;
  uint32_t strideW = 1;
  uint32_t dilationD = 1;
  uint32_t dilationH = 1;
  uint32_t dilationW = 1;
  uint32_t padFront = 0;
  uint32_t padTop = 0;
  uint32_t padLeft = 0;
  uint32_t padBack = 0;
  uint32_t padBottom = 0;
  uint32_t padRight = 0;
  uint32_t outD = 0;
  uint32_t outH = 0;
  uint32_t outW = 0;
};

enum class EltwiseKind {
  Add,
  Sub,
  Mul,
  Div,
  Pow,
  Mod,
  FloorMod,
  Prelu,
  SquaredDiff,
  Min,
  Max,
  LogicalAnd,
  LogicalOr,
  LogicalXor,
  Equal,
  NotEqual,
  Less,
  Greater,
  LessEqual,
  GreaterEqual
};

enum class ReduceKind { Sum, Mean, Max, Min, Prod, L1, L2, LogicalAnd, LogicalOr };

enum class TopKSortType { None = 0, SortValues = 1, SortIndices = 2 };

struct EltwiseCodegenDesc : BaseCodegenDesc {
  ov::element::Type input0_type{ov::element::dynamic};
  ov::element::Type input1_type{ov::element::dynamic};
  ov::element::Type output_type{ov::element::dynamic};
  EltwiseKind eltwise_kind{EltwiseKind::Add};
  uint32_t num_elements = 0;
  bool is_broadcast = false;
  bool use_half_compute = false;
  std::vector<int64_t> out_shape;
  std::vector<int64_t> stride0;
  std::vector<int64_t> stride1;
  bool has_activation = false;
  ActivationKind activation = ActivationKind::Relu;
  float alpha = 0.0f;
  bool has_input_activation = false;
  uint32_t input_activation_index = 0;
  ActivationKind input_activation = ActivationKind::Relu;
  float input_activation_alpha = 0.0f;
};

struct Pool2DCodegenDesc : BaseCodegenDesc {
  uint32_t N = 0;
  uint32_t C = 0;
  uint32_t H = 0;
  uint32_t W = 0;
  uint32_t kH = 0;
  uint32_t kW = 0;
  uint32_t strideH = 1;
  uint32_t strideW = 1;
  uint32_t dilationH = 1;
  uint32_t dilationW = 1;
  uint32_t padTop = 0;
  uint32_t padLeft = 0;
  uint32_t padBottom = 0;
  uint32_t padRight = 0;
  uint32_t outH = 0;
  uint32_t outW = 0;
  bool is_avg = false;
  bool exclude_pad = true;
};

struct TopKCodegenDesc : BaseCodegenDesc {
  ov::element::Type index_type{ov::element::i32};
  uint32_t axis_len = 0;
  uint32_t k = 0;
  uint32_t outer = 1;
  uint32_t inner = 1;
  bool mode_max = true;
  TopKSortType sort_type = TopKSortType::SortValues;
};

struct SoftmaxCodegenDesc : BaseCodegenDesc {
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t inner = 1;
  bool log_softmax = false;
};

struct BatchNorm2DCodegenDesc : BaseCodegenDesc {
  uint32_t N = 0;
  uint32_t C = 0;
  uint32_t H = 0;
  uint32_t W = 0;
};

struct UnaryCodegenDesc : BaseCodegenDesc {
  ActivationKind activation = ActivationKind::Relu;
  std::string entry_point = "unary_kernel";
  float alpha = 0.0f;
  double clamp_min = 0.0;
  double clamp_max = 0.0;
  bool gelu_tanh_approximation = false;
};

struct InterpolateCodegenDesc : BaseCodegenDesc {
  uint32_t N = 0;
  uint32_t C = 0;
  uint32_t H_in = 0;
  uint32_t W_in = 0;
  uint32_t H_out = 0;
  uint32_t W_out = 0;
  float scale_h = 1.f;
  float scale_w = 1.f;
  bool align_corners = false;
  bool nearest = true; // false → bilinear
  bool use_half_pixel = true;
  uint32_t nearest_mode = 0; // 0: round, 1: floor, 2: ceil
};

struct SplitCodegenDesc : BaseCodegenDesc {
  int64_t axis = 0;
  uint64_t inner = 1;
  uint64_t outer = 1;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> source_input_shape;
  std::vector<int64_t> input_permutation;
  std::vector<size_t> split_sizes;
};

struct TransposeCodegenDesc : BaseCodegenDesc {
  std::vector<uint32_t> in_shape;
  std::vector<uint32_t> out_shape;
  std::vector<uint32_t> perm;
  bool use_half = false;
  bool use_int = false;
};

struct ConcatCodegenDesc : BaseCodegenDesc {
  uint64_t outer = 0;
  uint64_t inner = 0;
  uint64_t axis_offset = 0;
  uint64_t axis_len = 0;
  uint64_t axis_total = 0;
  std::vector<uint64_t> input_axis_lengths;
};

struct ConvertCodegenDesc : BaseCodegenDesc {
  ov::element::Type src_type{ov::element::dynamic};
  ov::element::Type dst_type{ov::element::dynamic};
};

struct GatherCodegenDesc : BaseCodegenDesc {
  uint64_t outer = 0;
  uint64_t inner = 0;
  uint64_t axis_dim = 0;
  uint64_t indices_count = 0;
  ov::element::Type index_type{ov::element::i64};
};

struct GatherNDCodegenDesc : BaseCodegenDesc {
  static constexpr size_t kMaxDims = 8;
  uint32_t inner = 0;
  uint32_t num_indices = 0;
  uint32_t k = 0;
  uint32_t total = 0;
  std::array<uint32_t, kMaxDims> strides{};
  std::array<uint32_t, kMaxDims> dims{};
  ov::element::Type index_type{ov::element::i64};
};

struct GatherElementsCodegenDesc : BaseCodegenDesc {
  static constexpr size_t kMaxDims = 8;
  static constexpr size_t kRankOffset = 0;
  static constexpr size_t kAxisOffset = 1;
  static constexpr size_t kTotalOffset = 2;
  static constexpr size_t kOutDimsOffset = 3;
  static constexpr size_t kOutStridesOffset = kOutDimsOffset + kMaxDims;
  static constexpr size_t kDataDimsOffset = kOutStridesOffset + kMaxDims;
  static constexpr size_t kDataStridesOffset = kDataDimsOffset + kMaxDims;
  static constexpr size_t kParamU32Count = kDataStridesOffset + kMaxDims;
  uint32_t rank = 0;
  uint32_t axis = 0;
  uint32_t total = 0;
  std::array<uint32_t, kMaxDims> out_dims{};
  std::array<uint32_t, kMaxDims> out_strides{};
  std::array<uint32_t, kMaxDims> data_dims{};
  std::array<uint32_t, kMaxDims> data_strides{};
  ov::element::Type index_type{ov::element::i64};
};

struct DepthToSpaceCodegenDesc : BaseCodegenDesc {
  uint32_t N = 0;
  uint32_t C = 0;
  uint32_t H = 0;
  uint32_t W = 0;
  uint32_t C_out = 0;
  uint32_t H_out = 0;
  uint32_t W_out = 0;
  uint32_t block = 1;
  uint32_t mode = 0; // 0: BLOCKS_FIRST, 1: DEPTH_FIRST
  uint32_t total = 0;
};

struct SpaceToDepthCodegenDesc : BaseCodegenDesc {
  uint32_t N = 0;
  uint32_t C = 0;
  uint32_t H = 0;
  uint32_t W = 0;
  uint32_t C_out = 0;
  uint32_t H_out = 0;
  uint32_t W_out = 0;
  uint32_t block = 1;
  uint32_t mode = 0; // 0: BLOCKS_FIRST, 1: DEPTH_FIRST
  uint32_t total = 0;
};

struct ScatterElementsUpdateCodegenDesc : BaseCodegenDesc {
  static constexpr size_t kMaxDims = 8;
  uint32_t rank = 0;
  uint32_t axis = 0;
  uint32_t total_updates = 0;
  uint32_t total_data = 0;
  std::array<uint32_t, kMaxDims> update_dims{};
  std::array<uint32_t, kMaxDims> update_strides{};
  std::array<uint32_t, kMaxDims> data_dims{};
  std::array<uint32_t, kMaxDims> data_strides{};
  ov::element::Type index_type{ov::element::i64};
};

struct ScatterUpdateCodegenDesc : BaseCodegenDesc {
  ov::element::Type index_type{ov::element::i64};
};

struct ScatterNDUpdateCodegenDesc : BaseCodegenDesc {
  static constexpr size_t kMaxDims = 8;
  uint32_t inner = 0;
  uint32_t num_indices = 0;
  uint32_t k = 0;
  uint32_t total_updates = 0;
  uint32_t total_data = 0;
  std::array<uint32_t, kMaxDims> strides{};
  std::array<uint32_t, kMaxDims> dims{};
  ov::element::Type index_type{ov::element::i64};
};

struct ShapeOfCodegenDesc : BaseCodegenDesc {
  uint32_t rank = 0;
};

struct ReduceCodegenDesc : BaseCodegenDesc {
  ReduceKind kind{ReduceKind::Sum};
};

struct RmsCodegenDesc : BaseCodegenDesc {
  ov::element::Type input_type{ov::element::dynamic};
  ov::element::Type gamma_type{ov::element::dynamic};
  ov::element::Type output_type{ov::element::dynamic};
  uint32_t hidden = 0;
  uint32_t gamma_size = 0;
  uint32_t reduction_threads = 1;
  float epsilon = 0.0f;
  bool has_residual_add = false;
};

struct RopeCodegenDesc : BaseCodegenDesc {
  ov::element::Type input_type{ov::element::dynamic};
  ov::element::Type cos_type{ov::element::dynamic};
  ov::element::Type sin_type{ov::element::dynamic};
  ov::element::Type output_type{ov::element::dynamic};
  ov::element::Type position_type{ov::element::dynamic};
  uint32_t rank = 0;
  uint32_t batch = 1;
  uint32_t heads = 1;
  uint32_t head_size = 0;
  uint32_t rotary_dims = 0;
  uint32_t cos_sin_dims = 0;
  uint32_t cos_rank = 0;
  std::array<uint32_t, 4> cos_dims{{1, 1, 1, 1}};
  uint32_t cos_dynamic_mask = 0;
  bool is_interleaved = false;
  bool input_trans0213 = false;
  bool output_trans0213 = false;
  bool has_position = false;
};

inline uint32_t gfx_rms_parallel_reduction_threads(uint32_t hidden) {
  if (hidden >= 1024) {
    return 256u;
  }
  if (hidden >= 256) {
    return 128u;
  }
  if (hidden >= 64) {
    return 64u;
  }
  return 1u;
}

struct PadCodegenDesc : BaseCodegenDesc {
  double pad_value = 0.0;
};

struct TileCodegenDesc : BaseCodegenDesc {};
struct BroadcastCodegenDesc : BaseCodegenDesc {
  bool has_target_shape_input = false;
};
struct RangeCodegenDesc : BaseCodegenDesc {
  ov::element::Type start_type{ov::element::dynamic};
  ov::element::Type stop_type{ov::element::dynamic};
  ov::element::Type step_type{ov::element::dynamic};
  ov::element::Type output_type{ov::element::dynamic};
};
struct ReverseCodegenDesc : BaseCodegenDesc {
  static constexpr size_t kMaxDims = 8;
  uint32_t rank = 0;
  uint32_t total = 0;
  uint32_t axes_mask = 0;
  std::array<uint32_t, kMaxDims> dims{};
  std::array<uint32_t, kMaxDims> strides{};
};

} // namespace gfx_plugin
} // namespace ov
