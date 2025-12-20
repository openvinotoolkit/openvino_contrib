// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "runtime/metal_op_kinds.hpp"
#include "openvino/core/type/element_type.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"

namespace ov {
namespace metal_plugin {

struct BaseCodegenDesc {
    ov::element::Type element_type = ov::element::f32;
};

// Utility helpers to extract types from MLIR modules for lightweight codegen.
inline mlir::func::FuncOp get_entry_func(mlir::ModuleOp module) {
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
        return func;
    }
    return {};
}

inline mlir::Type get_tensor_element_type(mlir::Type type) {
    if (auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
        return ranked.getElementType();
    }
    if (auto unranked = llvm::dyn_cast<mlir::UnrankedTensorType>(type)) {
        return unranked.getElementType();
    }
    return type;
}

inline std::string msl_type_from_mlir(mlir::Type type) {
    type = get_tensor_element_type(type);
    if (llvm::isa<mlir::Float16Type>(type)) {
        return "half";
    }
    if (llvm::isa<mlir::Float32Type>(type)) {
        return "float";
    }
    if (llvm::isa<mlir::IndexType>(type)) {
        return "uint";
    }
    if (auto it = llvm::dyn_cast<mlir::IntegerType>(type)) {
        const bool is_signed = it.isSigned() || it.isSignless();
        switch (it.getWidth()) {
            case 1: return "bool";
            case 8: return is_signed ? "char" : "uchar";
            case 16: return is_signed ? "short" : "ushort";
            case 32: return is_signed ? "int" : "uint";
            case 64: return is_signed ? "long" : "ulong";
            default: return is_signed ? "int" : "uint";
        }
    }
    return "float";
}

inline std::string msl_type_from_element(const ov::element::Type& type) {
    if (type == ov::element::f16) {
        return "half";
    }
    if (type == ov::element::f32) {
        return "float";
    }
    if (type == ov::element::i8) {
        return "char";
    }
    if (type == ov::element::u8) {
        return "uchar";
    }
    if (type == ov::element::i16) {
        return "short";
    }
    if (type == ov::element::u16) {
        return "ushort";
    }
    if (type == ov::element::i32) {
        return "int";
    }
    if (type == ov::element::u32) {
        return "uint";
    }
    if (type == ov::element::i64) {
        return "long";
    }
    if (type == ov::element::u64) {
        return "ulong";
    }
    if (type == ov::element::boolean) {
        return "bool";
    }
    return "float";
}

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

// Per-op emitters (MSL generation; MLIR module is used when pattern-based codegen is available).
std::string generate_msl_for_matmul(const MatMulCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_conv2d(const Conv2DCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_conv3d(const Conv3DCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_eltwise(const EltwiseCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_maxpool2d(const Pool2DCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_avgpool2d(const Pool2DCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_softmax(const SoftmaxCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_concat(const ConcatCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_interpolate(const InterpolateCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_split(const SplitCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_transpose(const TransposeCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_convert(const ConvertCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_slice_generic(const ConvertCodegenDesc& desc, mlir::ModuleOp module); // reuse ConvertCodegenDesc for dtype only
std::string generate_msl_for_gather(const GatherCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_gathernd(const GatherNDCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_gather_elements(const GatherElementsCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_depth_to_space(const DepthToSpaceCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_space_to_depth(const SpaceToDepthCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_scatter_elements_update(const ScatterElementsUpdateCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_scatter_nd_update(const ScatterNDUpdateCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_shapeof(const ShapeOfCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_batchnorm2d(const BatchNorm2DCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_unary(const UnaryCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_select(mlir::ModuleOp module, ov::element::Type et);
std::string generate_msl_for_reduce(const ReduceCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_pad(const PadCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_tile(const TileCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_broadcast(const BroadcastCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_range(const RangeCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_reverse(const ReverseCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_topk(const TopKCodegenDesc& desc, mlir::ModuleOp module);

// Per-op wrappers to keep call sites uniform without central dispatch.
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const MatMulCodegenDesc& desc) {
    return generate_msl_for_matmul(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const Conv2DCodegenDesc& desc) {
    return generate_msl_for_conv2d(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const Conv3DCodegenDesc& desc) {
    return generate_msl_for_conv3d(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const EltwiseCodegenDesc& desc) {
    return generate_msl_for_eltwise(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const Pool2DCodegenDesc& desc) {
    if (desc.is_avg) {
        return generate_msl_for_avgpool2d(desc, module);
    }
    return generate_msl_for_maxpool2d(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const SoftmaxCodegenDesc& desc) {
    return generate_msl_for_softmax(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const UnaryCodegenDesc& desc) {
    return generate_msl_for_unary(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const BatchNorm2DCodegenDesc& desc) {
    return generate_msl_for_batchnorm2d(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const ConcatCodegenDesc& desc) {
    return generate_msl_for_concat(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const InterpolateCodegenDesc& desc) {
    return generate_msl_for_interpolate(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const SplitCodegenDesc& desc) {
    return generate_msl_for_split(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const TransposeCodegenDesc& desc) {
    return generate_msl_for_transpose(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const ConvertCodegenDesc& desc) {
    return generate_msl_for_convert(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const GatherCodegenDesc& desc) {
    return generate_msl_for_gather(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const GatherNDCodegenDesc& desc) {
    return generate_msl_for_gathernd(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const GatherElementsCodegenDesc& desc) {
    return generate_msl_for_gather_elements(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const DepthToSpaceCodegenDesc& desc) {
    return generate_msl_for_depth_to_space(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const SpaceToDepthCodegenDesc& desc) {
    return generate_msl_for_space_to_depth(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const ScatterElementsUpdateCodegenDesc& desc) {
    return generate_msl_for_scatter_elements_update(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const ScatterNDUpdateCodegenDesc& desc) {
    return generate_msl_for_scatter_nd_update(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const ShapeOfCodegenDesc& desc) {
    return generate_msl_for_shapeof(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, ov::element::Type et, int /*select_tag*/) {
    return generate_msl_for_select(module, et);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const ReduceCodegenDesc& desc) {
    return generate_msl_for_reduce(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const PadCodegenDesc& desc) {
    return generate_msl_for_pad(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const TileCodegenDesc& desc) {
    return generate_msl_for_tile(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const BroadcastCodegenDesc& desc) {
    return generate_msl_for_broadcast(desc, module);
}
inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const RangeCodegenDesc& desc) {
    return generate_msl_for_range(desc, module);
}

inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const ReverseCodegenDesc& desc) {
    return generate_msl_for_reverse(desc, module);
}

inline std::string generate_msl_from_mlir(mlir::ModuleOp module, const TopKCodegenDesc& desc) {
    return generate_msl_for_topk(desc, module);
}

}  // namespace metal_plugin
}  // namespace ov
