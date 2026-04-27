// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/parallel_fill_fusion.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace ov {
namespace gfx_plugin {
namespace {

mlir::Value strip_memref_casts(mlir::Value value) {
    mlir::Value current = value;
    while (current) {
        if (auto cast = current.getDefiningOp<mlir::memref::CastOp>()) {
            current = cast.getSource();
            continue;
        }
        if (auto subview = current.getDefiningOp<mlir::memref::SubViewOp>()) {
            current = subview.getSource();
            continue;
        }
        if (auto view = current.getDefiningOp<mlir::memref::ViewOp>()) {
            current = view.getSource();
            continue;
        }
        break;
    }
    return current;
}

bool is_zero_constant(mlir::Value value) {
    if (auto cst = value.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto fattr = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
            return fattr.getValueAsDouble() == 0.0;
        }
        if (auto iattr = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue())) {
            return iattr.getInt() == 0;
        }
    }
    return false;
}

mlir::Value make_zero(mlir::OpBuilder& b, mlir::Location loc, mlir::Type elem_ty) {
    if (auto fty = mlir::dyn_cast<mlir::FloatType>(elem_ty)) {
        return b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(fty, 0.0)).getResult();
    }
    if (auto ity = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
        return b.create<mlir::arith::ConstantOp>(loc, b.getIntegerAttr(ity, 0)).getResult();
    }
    return b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(b.getF32Type(), 0.0)).getResult();
}

bool same_parallel_bounds(mlir::scf::ParallelOp a, mlir::scf::ParallelOp b) {
    if (a.getNumLoops() != b.getNumLoops()) {
        return false;
    }
    for (unsigned i = 0; i < a.getNumLoops(); ++i) {
        if (a.getLowerBound()[i] != b.getLowerBound()[i]) {
            return false;
        }
        if (a.getUpperBound()[i] != b.getUpperBound()[i]) {
            return false;
        }
        if (a.getStep()[i] != b.getStep()[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace

void run_parallel_fill_fusion(mlir::ModuleOp module) {
    if (!module) {
        return;
    }
    module.walk([&](mlir::func::FuncOp func) {
        for (auto it = func.getBody().begin(); it != func.getBody().end(); ++it) {
            auto* block = &*it;
            for (auto op_it = block->begin(); op_it != block->end(); ) {
                auto* op = &*op_it++;
                auto fill_par = mlir::dyn_cast<mlir::scf::ParallelOp>(op);
                if (!fill_par) {
                    continue;
                }
                auto* next_op = op->getNextNode();
                auto compute_par = mlir::dyn_cast_or_null<mlir::scf::ParallelOp>(next_op);
                if (!compute_par) {
                    continue;
                }
                if (!same_parallel_bounds(fill_par, compute_par)) {
                    continue;
                }

                // Find zero fill store in the first parallel op.
                mlir::memref::StoreOp fill_store;
                for (auto& inner_op : fill_par.getBody()->getOperations()) {
                    if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(&inner_op)) {
                        if (fill_store) {
                            fill_store = nullptr;
                            break;
                        }
                        if (!is_zero_constant(store.getValue())) {
                            fill_store = nullptr;
                            break;
                        }
                        fill_store = store;
                    }
                }
                if (!fill_store) {
                    continue;
                }

                auto fill_mem = strip_memref_casts(fill_store.getMemRef());

                // Find a store to the same memref in the compute loop.
                mlir::memref::StoreOp compute_store;
                compute_par.walk([&](mlir::memref::StoreOp store) {
                    if (!compute_store &&
                        strip_memref_casts(store.getMemRef()) == fill_mem) {
                        compute_store = store;
                    }
                });
                if (!compute_store) {
                    continue;
                }

                // Insert zero store at the start of compute parallel body.
                auto loc = compute_par.getLoc();
                mlir::OpBuilder b(compute_par.getBody(), compute_par.getBody()->begin());
                mlir::Type elem_ty = compute_store.getValue().getType();
                auto zero = make_zero(b, loc, elem_ty);
                b.create<mlir::memref::StoreOp>(loc,
                                               zero,
                                               compute_store.getMemRef(),
                                               compute_store.getIndices());

                fill_par.erase();
                // Restart scan after mutation.
                op_it = block->begin();
                break;
            }
        }
    });
}

}  // namespace gfx_plugin
}  // namespace ov
