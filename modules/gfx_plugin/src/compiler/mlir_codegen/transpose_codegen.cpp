// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "mlir_codegen/codegen_common.hpp"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"

#include <sstream>
#include <utility>
#include <vector>

#include "llvm/Support/Casting.h"

namespace ov {
namespace gfx_plugin {

namespace {
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

bool get_perm_from_linalg(mlir::ModuleOp module, std::vector<uint32_t>& perm_out) {
    auto func = get_entry_func(module);
    if (!func)
        return false;
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
    const unsigned rank = map_in.getNumResults();
    std::vector<int64_t> inv_perm(rank, -1);
    for (unsigned i = 0; i < rank; ++i) {
        auto expr = llvm::dyn_cast<mlir::AffineDimExpr>(map_in.getResult(i));
        if (!expr)
            return false;
        inv_perm[i] = static_cast<int64_t>(expr.getPosition());
    }
    perm_out.assign(rank, 0);
    for (unsigned i = 0; i < rank; ++i) {
        if (inv_perm[i] < 0 || static_cast<unsigned>(inv_perm[i]) >= rank)
            return false;
        perm_out[static_cast<size_t>(inv_perm[i])] = i;
    }
    return true;
}
}  // namespace

std::string generate_msl_for_transpose(const TransposeCodegenDesc& d, mlir::ModuleOp module) {
    std::string scalar_ty = "float";
    std::vector<uint32_t> out_shape = d.out_shape;
    std::vector<uint32_t> in_shape = d.in_shape;
    std::vector<uint32_t> perm = d.perm;

    if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumInputs() >= 1) {
            scalar_ty = msl_type_from_mlir(ft.getInput(0));
        }
        if (ft.getNumInputs() >= 1) {
            if (auto rt = llvm::dyn_cast<mlir::RankedTensorType>(ft.getInput(0))) {
                if (rt.hasStaticShape()) {
                    in_shape.clear();
                    for (auto dim : rt.getShape())
                        in_shape.push_back(static_cast<uint32_t>(dim));
                }
            }
        }
        if (ft.getNumResults() >= 1) {
            if (auto rt = llvm::dyn_cast<mlir::RankedTensorType>(ft.getResult(0))) {
                if (rt.hasStaticShape()) {
                    out_shape.clear();
                    for (auto dim : rt.getShape())
                        out_shape.push_back(static_cast<uint32_t>(dim));
                }
            }
        }
        std::vector<uint32_t> mlir_perm;
        if (get_perm_from_linalg(module, mlir_perm)) {
            perm = std::move(mlir_perm);
        }
    } else {
        if (d.use_half) scalar_ty = "half";
        else if (d.use_int) scalar_ty = "int";
    }

    const uint32_t rank = static_cast<uint32_t>(out_shape.size());
    std::vector<uint32_t> in_stride = compute_stride(in_shape);

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\nusing namespace metal;\n";
    ss << "constant uint RANK = " << rank << ";\n";
    ss << "constant uint OUT_SHAPE[" << rank << "] = {";
    for (size_t i = 0; i < out_shape.size(); ++i) {
        if (i) ss << ", ";
        ss << out_shape[i];
    }
    ss << "};\n";
    ss << "constant uint PERM[" << rank << "] = {";
    for (size_t i = 0; i < perm.size(); ++i) {
        if (i) ss << ", ";
        ss << perm[i];
    }
    ss << "};\n";
    ss << "constant uint IN_STRIDE[" << rank << "] = {";
    for (size_t i = 0; i < in_stride.size(); ++i) {
        if (i) ss << ", ";
        ss << in_stride[i];
    }
    ss << "};\n";
    ss << "kernel void transpose_kernel(\n";
    ss << "  device const " << scalar_ty << "* A [[buffer(0)]],\n";
    ss << "  device " << scalar_ty << "* C [[buffer(1)]],\n";
    ss << "  constant uint& NUM_ELEMS [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= NUM_ELEMS) return;\n";
    ss << "    uint idx = gid;\n";
    ss << "    uint off_in = 0;\n";
    ss << "    for (int d = (int)RANK - 1; d >= 0; --d) {\n";
    ss << "        uint coord = idx % OUT_SHAPE[d];\n";
    ss << "        idx /= OUT_SHAPE[d];\n";
    ss << "        uint p = PERM[d];\n";
    ss << "        off_in += coord * IN_STRIDE[p];\n";
    ss << "    }\n";
    ss << "    C[gid] = A[off_in];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
