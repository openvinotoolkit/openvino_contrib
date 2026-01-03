// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "runtime/gfx_activation.hpp"

namespace ov {
namespace gfx_plugin {

struct BaseCodegenDesc {
    ov::element::Type element_type = ov::element::f32;
};

struct MatMulCodegenDesc : BaseCodegenDesc {
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
    std::array<int64_t, 3> bias_dims{{1, 1, 1}};  // aligned to [batch, M, N]
    bool has_activation = false;
    ActivationKind activation = ActivationKind::Relu;
    float alpha = 0.0f;
};

struct Conv2DCodegenDesc : BaseCodegenDesc {
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
    uint32_t bias_rank = 1;  // 1 or 4
    bool has_activation = false;
    ActivationKind activation = ActivationKind::Relu;
    float alpha = 0.0f;
    bool use_special_k3 = false;  // enable k=3 stride1/2 optimized kernel
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

enum class ReduceKind { Sum, Mean, Max, Min, Prod, L1, L2 };

enum class TopKSortType {
    None = 0,
    SortValues = 1,
    SortIndices = 2
};

struct EltwiseCodegenDesc : BaseCodegenDesc {
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
    float alpha = 0.0f;
    double clamp_min = 0.0;
    double clamp_max = 0.0;
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
    bool nearest = true;  // false → bilinear
    bool use_half_pixel = true;
    uint32_t nearest_mode = 0;  // 0: round, 1: floor, 2: ceil
};

struct SplitCodegenDesc : BaseCodegenDesc {
    int64_t axis = 0;
    uint64_t inner = 1;
    uint64_t outer = 1;
    std::vector<int64_t> input_shape;
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
    uint32_t mode = 0;  // 0: BLOCKS_FIRST, 1: DEPTH_FIRST
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
    uint32_t mode = 0;  // 0: BLOCKS_FIRST, 1: DEPTH_FIRST
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

struct PadCodegenDesc : BaseCodegenDesc {
    double pad_value = 0.0;
};

struct TileCodegenDesc : BaseCodegenDesc {};
struct BroadcastCodegenDesc : BaseCodegenDesc {};
struct RangeCodegenDesc : BaseCodegenDesc {};
struct ReverseCodegenDesc : BaseCodegenDesc {
    static constexpr size_t kMaxDims = 8;
    uint32_t rank = 0;
    uint32_t total = 0;
    uint32_t axes_mask = 0;
    std::array<uint32_t, kMaxDims> dims{};
    std::array<uint32_t, kMaxDims> strides{};
};

}  // namespace gfx_plugin
}  // namespace ov
