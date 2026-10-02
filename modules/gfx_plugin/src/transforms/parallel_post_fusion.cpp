// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/parallel_post_fusion.hpp"

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

bool find_single_store(mlir::scf::ParallelOp loop, mlir::memref::StoreOp& out_store) {
    out_store = nullptr;
    loop.walk([&](mlir::memref::StoreOp store) {
        if (!out_store) {
            out_store = store;
            return;
        }
        out_store = nullptr;
    });
    return static_cast<bool>(out_store);
}

}  // namespace

void run_parallel_post_fusion(mlir::ModuleOp module) {
    if (!module) {
        return;
    }
    module.walk([&](mlir::func::FuncOp func) {
        for (auto it = func.getBody().begin(); it != func.getBody().end(); ++it) {
            auto* block = &*it;
            for (auto op_it = block->begin(); op_it != block->end(); ) {
                auto* op = &*op_it++;
                auto compute_par = mlir::dyn_cast<mlir::scf::ParallelOp>(op);
                if (!compute_par) {
                    continue;
                }
                auto* next_op = op->getNextNode();
                auto post_par = mlir::dyn_cast_or_null<mlir::scf::ParallelOp>(next_op);
                if (!post_par) {
                    continue;
                }
                if (!same_parallel_bounds(compute_par, post_par)) {
                    continue;
                }

                mlir::memref::StoreOp post_store;
                if (!find_single_store(post_par, post_store)) {
                    continue;
                }
                mlir::Value post_memref = strip_memref_casts(post_store.getMemRef());

                // Ensure compute loop writes to the same memref.
                mlir::memref::StoreOp compute_store;
                compute_par.walk([&](mlir::memref::StoreOp store) {
                    if (!compute_store && strip_memref_casts(store.getMemRef()) == post_memref) {
                        compute_store = store;
                    }
                });
                if (!compute_store) {
                    continue;
                }

                // Clone post loop body into compute loop body with IV remapping.
                mlir::IRMapping mapping;
                auto post_ivs = post_par.getInductionVars();
                auto compute_ivs = compute_par.getInductionVars();
                if (post_ivs.size() != compute_ivs.size()) {
                    continue;
                }
                for (size_t i = 0; i < post_ivs.size(); ++i) {
                    mapping.map(post_ivs[i], compute_ivs[i]);
                }

                auto* term = compute_par.getBody()->getTerminator();
                mlir::OpBuilder b(term);
                auto* post_block = post_par.getBody();
                auto* post_term = post_block->getTerminator();
                for (auto& body_op : post_block->getOperations()) {
                    if (&body_op == post_term) {
                        break;
                    }
                    b.clone(body_op, mapping);
                }

                post_par.erase();
                op_it = block->begin();
                break;
            }
        }
    });
}

}  // namespace gfx_plugin
}  // namespace ov
