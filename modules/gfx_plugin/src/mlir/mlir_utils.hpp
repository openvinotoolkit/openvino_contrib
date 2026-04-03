// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

namespace ov {
namespace gfx_plugin {

inline bool strip_strided_func_layouts(mlir::ModuleOp module, bool log_debug) {
    bool updated = false;
    module.walk([&](mlir::func::FuncOp func) {
        if (func.isExternal()) {
            return;
        }
        auto fn_type = func.getFunctionType();
        llvm::SmallVector<mlir::Type, 8> inputs;
        inputs.reserve(fn_type.getNumInputs());

        bool changed = false;
        for (auto type : fn_type.getInputs()) {
            if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type)) {
                auto plain = mlir::MemRefType::get(memref.getShape(),
                                                   memref.getElementType(),
                                                   mlir::AffineMap(),
                                                   memref.getMemorySpace());
                inputs.push_back(plain);
                changed |= (plain != memref);
            } else {
                inputs.push_back(type);
            }
        }

        if (!changed) {
            return;
        }

        auto new_type = mlir::FunctionType::get(func.getContext(), inputs, fn_type.getResults());
        func.setType(new_type);
        auto& entry = func.getBody().front();
        for (size_t i = 0; i < inputs.size(); ++i) {
            entry.getArgument(static_cast<unsigned>(i)).setType(inputs[i]);
        }
        updated = true;
    });

    if (updated && log_debug) {
        llvm::errs() << "[GFX][MLIR] Stripped strided layouts from func arguments\n";
    }
    return updated;
}

}  // namespace gfx_plugin
}  // namespace ov
