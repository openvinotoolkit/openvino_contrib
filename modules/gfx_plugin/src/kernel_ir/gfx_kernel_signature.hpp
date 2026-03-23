// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace ov {
namespace gfx_plugin {

struct KernelFunctionSignature {
    uint32_t inputs = 0;
    uint32_t results = 0;

    uint32_t total() const { return inputs + results; }
};

inline KernelFunctionSignature infer_kernel_signature(mlir::ModuleOp module,
                                                      const std::string& entry_point) {
    if (!module) {
        return {};
    }
    mlir::func::FuncOp func;
    if (!entry_point.empty()) {
        func = module.lookupSymbol<mlir::func::FuncOp>(entry_point);
    }
    if (!func) {
        module.walk([&](mlir::func::FuncOp f) {
            if (!func) {
                func = f;
            }
        });
    }
    if (!func) {
        return {};
    }
    auto ftype = func.getFunctionType();
    KernelFunctionSignature sig;
    sig.inputs = static_cast<uint32_t>(ftype.getNumInputs());
    sig.results = static_cast<uint32_t>(ftype.getNumResults());
    return sig;
}

}  // namespace gfx_plugin
}  // namespace ov
