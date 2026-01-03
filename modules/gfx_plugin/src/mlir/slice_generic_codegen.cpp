// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"

#include <sstream>
#include <vector>

#include "llvm/Support/Casting.h"

namespace ov {
namespace gfx_plugin {

namespace {
struct SliceMeta {
    std::vector<uint32_t> out_shape;
    std::vector<uint32_t> in_stride;
    std::vector<int32_t> starts;
    std::vector<uint32_t> steps;
};

std::vector<uint32_t> compute_stride(const std::vector<uint32_t>& shape) {
    std::vector<uint32_t> stride(shape.size(), 1);
    if (shape.empty())
        return stride;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        stride[static_cast<size_t>(i)] =
            stride[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return stride;
}

bool parse_affine_dim_expr(mlir::AffineExpr expr, int64_t& dim_pos, int64_t& scale, int64_t& offset) {
    if (auto d = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
        dim_pos = d.getPosition();
        scale = 1;
        offset = 0;
        return true;
    }
    if (auto c = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
        dim_pos = -1;
        scale = 0;
        offset = c.getValue();
        return true;
    }
    if (auto bin = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
        auto lhs = bin.getLHS();
        auto rhs = bin.getRHS();
        if (bin.getKind() == mlir::AffineExprKind::Add) {
            int64_t dim_l = -1, scale_l = 0, off_l = 0;
            int64_t dim_r = -1, scale_r = 0, off_r = 0;
            if (!parse_affine_dim_expr(lhs, dim_l, scale_l, off_l) ||
                !parse_affine_dim_expr(rhs, dim_r, scale_r, off_r)) {
                return false;
            }
            if (dim_l >= 0 && dim_r < 0) {
                dim_pos = dim_l;
                scale = scale_l;
                offset = off_l + off_r;
                return true;
            }
            if (dim_r >= 0 && dim_l < 0) {
                dim_pos = dim_r;
                scale = scale_r;
                offset = off_l + off_r;
                return true;
            }
            return false;
        }
        if (bin.getKind() == mlir::AffineExprKind::Mul) {
            int64_t dim_l = -1, scale_l = 0, off_l = 0;
            int64_t dim_r = -1, scale_r = 0, off_r = 0;
            if (!parse_affine_dim_expr(lhs, dim_l, scale_l, off_l) ||
                !parse_affine_dim_expr(rhs, dim_r, scale_r, off_r)) {
                return false;
            }
            if (dim_l >= 0 && dim_r < 0 && scale_r == 0) {
                dim_pos = dim_l;
                scale = scale_l * off_r;
                offset = 0;
                return true;
            }
            if (dim_r >= 0 && dim_l < 0 && scale_l == 0) {
                dim_pos = dim_r;
                scale = scale_r * off_l;
                offset = 0;
                return true;
            }
            return false;
        }
    }
    return false;
}

bool extract_slice_meta(mlir::ModuleOp module, SliceMeta& meta) {
    auto func = get_entry_func(module);
    if (!func)
        return false;
    auto ft = func.getFunctionType();
    if (ft.getNumInputs() < 1 || ft.getNumResults() < 1)
        return false;
    auto in_ty = llvm::dyn_cast<mlir::RankedTensorType>(ft.getInput(0));
    auto out_ty = llvm::dyn_cast<mlir::RankedTensorType>(ft.getResult(0));
    if (!in_ty || !out_ty || !in_ty.hasStaticShape() || !out_ty.hasStaticShape())
        return false;

    const auto in_shape = in_ty.getShape();
    const auto out_shape = out_ty.getShape();
    meta.out_shape.clear();
    meta.out_shape.reserve(out_shape.size());
    for (auto d : out_shape)
        meta.out_shape.push_back(static_cast<uint32_t>(d));

    std::vector<uint32_t> in_shape_u;
    in_shape_u.reserve(in_shape.size());
    for (auto d : in_shape)
        in_shape_u.push_back(static_cast<uint32_t>(d));
    meta.in_stride = compute_stride(in_shape_u);
    meta.starts.assign(out_shape.size(), 0);
    meta.steps.assign(out_shape.size(), 1);

    mlir::linalg::GenericOp generic = nullptr;
    func.walk([&](mlir::linalg::GenericOp op) {
        if (!generic)
            generic = op;
    });
    if (!generic)
        return false;

    auto maps = generic.getIndexingMapsArray();
    if (maps.size() < 2)
        return false;
    auto map_in = maps[0];
    if (map_in.getNumResults() != out_shape.size())
        return false;

    for (unsigned i = 0; i < map_in.getNumResults(); ++i) {
        int64_t dim_pos = -1, scale = 1, offset = 0;
        if (!parse_affine_dim_expr(map_in.getResult(i), dim_pos, scale, offset))
            return false;
        if (dim_pos < 0 || static_cast<size_t>(dim_pos) >= out_shape.size())
            return false;
        meta.starts[i] = static_cast<int32_t>(offset);
        meta.steps[i] = static_cast<uint32_t>(scale);
    }
    return true;
}
}  // namespace

// Uses ConvertCodegenDesc just to carry dst_type (dtype of slice tensors).
std::string generate_msl_for_slice_generic(const ConvertCodegenDesc& d, mlir::ModuleOp module) {
    std::string scalar_t = "float";
    if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumInputs() >= 1) {
            scalar_t = msl_type_from_mlir(ft.getInput(0));
        }
    } else {
        switch (d.dst_type) {
            case ov::element::f16: scalar_t = "half"; break;
            case ov::element::f32: scalar_t = "float"; break;
            case ov::element::i32: scalar_t = "int"; break;
            case ov::element::i64: scalar_t = "long"; break;
            default: break;
        }
    }
    SliceMeta meta{};
    const bool use_static = extract_slice_meta(module, meta);
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\nusing namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    if (use_static) {
        const uint32_t rank = static_cast<uint32_t>(meta.out_shape.size());
        ss << "constant uint RANK_C = " << rank << ";\n";
        ss << "constant uint OUT_SHAPE_C[" << rank << "] = {";
        for (size_t i = 0; i < meta.out_shape.size(); ++i) {
            if (i) ss << ", ";
            ss << meta.out_shape[i];
        }
        ss << "};\n";
        ss << "constant uint IN_STRIDE_C[" << rank << "] = {";
        for (size_t i = 0; i < meta.in_stride.size(); ++i) {
            if (i) ss << ", ";
            ss << meta.in_stride[i];
        }
        ss << "};\n";
        ss << "constant int STARTS_C[" << rank << "] = {";
        for (size_t i = 0; i < meta.starts.size(); ++i) {
            if (i) ss << ", ";
            ss << meta.starts[i];
        }
        ss << "};\n";
        ss << "constant uint STEPS_C[" << rank << "] = {";
        for (size_t i = 0; i < meta.steps.size(); ++i) {
            if (i) ss << ", ";
            ss << meta.steps[i];
        }
        ss << "};\n";
    }
    ss << "kernel void slice_kernel(\n";
    ss << "  device const scalar_t* A [[buffer(0)]],\n";
    ss << "  device scalar_t* C [[buffer(1)]],\n";
    ss << "  constant uint& TOTAL [[buffer(2)]],\n";
    ss << "  constant uint& RANK [[buffer(3)]],\n";
    ss << "  constant uint* out_shape [[buffer(4)]],\n";
    ss << "  constant uint* in_stride [[buffer(5)]],\n";
    ss << "  constant int* starts [[buffer(6)]],\n";
    ss << "  constant uint* steps [[buffer(7)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= TOTAL) return;\n";
    if (use_static) {
        ss << "    (void)RANK;\n";
        ss << "    (void)out_shape;\n";
        ss << "    (void)in_stride;\n";
        ss << "    (void)starts;\n";
        ss << "    (void)steps;\n";
    }
    ss << "    uint idx = gid;\n";
    ss << "    uint in_off = 0;\n";
    if (use_static) {
        ss << "    for (int d = (int)RANK_C - 1; d >= 0; --d) {\n";
        ss << "        uint coord = idx % OUT_SHAPE_C[d];\n";
        ss << "        idx /= OUT_SHAPE_C[d];\n";
        ss << "        in_off += (uint)((int)STARTS_C[d] + (int)(coord * STEPS_C[d])) * IN_STRIDE_C[d];\n";
    } else {
        ss << "    for (int d = (int)RANK - 1; d >= 0; --d) {\n";
        ss << "        uint coord = idx % out_shape[d];\n";
        ss << "        idx /= out_shape[d];\n";
        ss << "        in_off += (uint)((int)starts[d] + (int)(coord * steps[d])) * in_stride[d];\n";
    }
    ss << "    }\n";
    ss << "    C[gid] = A[in_off];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
