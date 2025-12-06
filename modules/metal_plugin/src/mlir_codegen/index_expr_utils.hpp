// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Value.h"

#ifndef METAL_MLIR_DEBUG
#define METAL_MLIR_DEBUG 0
#endif

namespace ov {
namespace metal_plugin {

inline void mlir_codegen_log(const std::string& msg) {
#if METAL_MLIR_DEBUG
    fprintf(stderr, "%s\n", msg.c_str());
#else
    if (std::getenv("METAL_MLIR_DEBUG")) {
        fprintf(stderr, "%s\n", msg.c_str());
    }
#endif
}

inline mlir::Value strip_memref_casts(mlir::Value v) {
    while (true) {
        if (auto cast = v.getDefiningOp<mlir::memref::CastOp>()) {
            v = cast.getSource();
            continue;
        }
        if (auto sub = v.getDefiningOp<mlir::memref::SubViewOp>()) {
            v = sub.getSource();
            continue;
        }
        break;
    }
    return v;
}

inline std::string render_affine_expr(mlir::AffineExpr expr, llvm::ArrayRef<std::string> syms) {
    if (auto d = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
        return syms[d.getPosition()];
    }
    if (auto s = llvm::dyn_cast<mlir::AffineSymbolExpr>(expr)) {
        return syms[s.getPosition()];
    }
    if (auto c = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
        return std::to_string(c.getValue());
    }
    if (auto a = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
        auto lhs = render_affine_expr(a.getLHS(), syms);
        auto rhs = render_affine_expr(a.getRHS(), syms);
        switch (expr.getKind()) {
            case mlir::AffineExprKind::Add:
                return "(" + lhs + " + " + rhs + ")";
            case mlir::AffineExprKind::Mul:
                return "(" + lhs + " * " + rhs + ")";
            case mlir::AffineExprKind::FloorDiv:
                return "(" + lhs + " floordiv " + rhs + ")";
            case mlir::AffineExprKind::CeilDiv:
                return "(" + lhs + " ceildiv " + rhs + ")";
            case mlir::AffineExprKind::Mod:
                return "(" + lhs + " mod " + rhs + ")";
            default:
                break;
        }
    }
    return "<affine?>";
}

inline std::string render_index_expr(mlir::Value v,
                                     const llvm::DenseMap<mlir::Value, std::string>& names,
                                     int depth = 0) {
    if (auto it = names.find(v); it != names.end())
        return it->second;
    if (depth > 32)
        return "<recursion>";
    if (auto c = v.getDefiningOp<mlir::arith::ConstantIndexOp>())
        return std::to_string(c.value());
    if (auto c = v.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
            return std::to_string(int_attr.getInt());
    }
    if (auto add = v.getDefiningOp<mlir::arith::AddIOp>()) {
        return "(" + render_index_expr(add.getLhs(), names, depth + 1) + " + " +
               render_index_expr(add.getRhs(), names, depth + 1) + ")";
    }
    if (auto sub = v.getDefiningOp<mlir::arith::SubIOp>()) {
        return "(" + render_index_expr(sub.getLhs(), names, depth + 1) + " - " +
               render_index_expr(sub.getRhs(), names, depth + 1) + ")";
    }
    if (auto mul = v.getDefiningOp<mlir::arith::MulIOp>()) {
        return "(" + render_index_expr(mul.getLhs(), names, depth + 1) + " * " +
               render_index_expr(mul.getRhs(), names, depth + 1) + ")";
    }
    if (auto divu = v.getDefiningOp<mlir::arith::DivUIOp>()) {
        return "(" + render_index_expr(divu.getLhs(), names, depth + 1) + " / " +
               render_index_expr(divu.getRhs(), names, depth + 1) + ")";
    }
    if (auto casts = v.getDefiningOp<mlir::arith::IndexCastOp>()) {
        return render_index_expr(casts.getIn(), names, depth + 1);
    }
    if (auto aff = v.getDefiningOp<mlir::affine::AffineApplyOp>()) {
        std::vector<std::string> syms;
        syms.reserve(aff.getMapOperands().size());
        for (auto op : aff.getMapOperands()) {
            syms.push_back(render_index_expr(op, names, depth + 1));
        }
        return render_affine_expr(aff.getAffineMap().getResult(0), syms);
    }
    return "<expr?>";
}

inline std::string flatten_indices(const std::vector<std::string>& idx,
                                   const std::vector<std::string>& dims) {
    if (idx.empty())
        return "0";
    std::string acc = idx[0];
    for (size_t i = 1; i < idx.size(); ++i) {
        const std::string& dim = dims.at(i - 1);
        acc = "(" + acc + " * " + dim + " + " + idx[i] + ")";
    }
    return acc;
}

}  // namespace metal_plugin
}  // namespace ov
