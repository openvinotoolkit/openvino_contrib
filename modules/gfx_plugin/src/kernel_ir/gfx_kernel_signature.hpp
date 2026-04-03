// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
    auto build_signature = [](mlir::FunctionType type) {
        KernelFunctionSignature sig;
        sig.inputs = static_cast<uint32_t>(type.getNumInputs());
        sig.results = static_cast<uint32_t>(type.getNumResults());
        return sig;
    };
    if (!entry_point.empty()) {
        if (auto func = module.lookupSymbol<mlir::func::FuncOp>(entry_point)) {
            return build_signature(func.getFunctionType());
        }
        if (auto gpu_func = module.lookupSymbol<mlir::gpu::GPUFuncOp>(entry_point)) {
            return build_signature(gpu_func.getFunctionType());
        }
    }
    mlir::gpu::GPUFuncOp gpu_func;
    module.walk([&](mlir::gpu::GPUFuncOp f) {
        if (!gpu_func) {
            gpu_func = f;
        }
    });
    if (gpu_func) {
        return build_signature(gpu_func.getFunctionType());
    }
    mlir::func::FuncOp func;
    module.walk([&](mlir::func::FuncOp f) {
        if (!func) {
            func = f;
        }
    });
    if (func) {
        return build_signature(func.getFunctionType());
    }
    return {};
}

}  // namespace gfx_plugin
}  // namespace ov
