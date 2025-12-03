// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_utils.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

#include <stdexcept>

namespace ov {
namespace metal_plugin {

void extract_matmul_shape(mlir::ModuleOp module, int64_t& M, int64_t& N, int64_t& K) {
    auto func = module.lookupSymbol<mlir::func::FuncOp>("matmul_main");
    if (!func) {
        throw std::runtime_error("matmul_main function not found in MLIR module");
    }

    auto arg_types = func.getFunctionType().getInputs();
    auto res_types = func.getFunctionType().getResults();

    if (arg_types.size() != 2 || res_types.size() != 1) {
        throw std::runtime_error("matmul_main signature is unexpected");
    }

    auto a_ty = mlir::dyn_cast<mlir::RankedTensorType>(arg_types[0]);
    auto b_ty = mlir::dyn_cast<mlir::RankedTensorType>(arg_types[1]);
    auto c_ty = mlir::dyn_cast<mlir::RankedTensorType>(res_types[0]);
    if (!a_ty || !b_ty || !c_ty) {
        throw std::runtime_error("matmul_main expects ranked tensor types");
    }
    if (a_ty.getRank() < 2 || b_ty.getRank() < 2 || c_ty.getRank() < 2 ||
        a_ty.getRank() > 3 || b_ty.getRank() > 3 || c_ty.getRank() > 3) {
        throw std::runtime_error("matmul_main supports only rank-2/3 tensors");
    }

    auto shape_a = a_ty.getShape();
    auto shape_b = b_ty.getShape();
    auto shape_c = c_ty.getShape();

    M = shape_a[a_ty.getRank() - 2];
    K = shape_a[a_ty.getRank() - 1];
    N = shape_b[b_ty.getRank() - 1];

    int64_t batch_a = (a_ty.getRank() == 3) ? shape_a[0] : 1;
    int64_t batch_b = (b_ty.getRank() == 3) ? shape_b[0] : 1;
    int64_t batch_c = (c_ty.getRank() == 3) ? shape_c[0] : 1;
    int64_t batch = std::max(batch_a, batch_b);

    if (shape_b[b_ty.getRank() - 2] != K || shape_c[c_ty.getRank() - 2] != M ||
        shape_c[c_ty.getRank() - 1] != N || batch_c != batch) {
        throw std::runtime_error("MatMul shapes mismatch inside MLIR module");
    }
}

}  // namespace metal_plugin
}  // namespace ov
