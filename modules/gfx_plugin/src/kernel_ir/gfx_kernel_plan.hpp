// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <functional>
#include <string>

#include "openvino/core/shape.hpp"
#include "kernel_ir/gfx_codegen_backend.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "kernel_ir/gfx_kernel_dispatch.hpp"

namespace ov {
namespace gfx_plugin {

uint32_t infer_kernel_arg_count(mlir::ModuleOp module, const std::string& entry_point);

class KernelPlan {
public:
    KernelPlan(mlir::ModuleOp module, std::string entry_point, uint32_t arg_count)
        : m_module(module),
          m_entry_point(std::move(entry_point)),
          m_arg_count(arg_count) {}

    mlir::ModuleOp module() const { return m_module; }
    const std::string& entry_point() const { return m_entry_point; }
    uint32_t arg_count() const { return m_arg_count; }

    KernelSource to_source() const {
        const uint32_t inferred = m_arg_count ? m_arg_count : infer_kernel_arg_count(m_module, m_entry_point);
        return make_kernel_source_from_mlir(m_module, m_entry_point, inferred);
    }

    KernelSource to_source_with_msl(std::string msl_source) const {
        KernelSource src = to_source();
        src.msl_source = std::move(msl_source);
        return src;
    }

    KernelSource to_source_with_msl_generator(std::function<std::string(mlir::ModuleOp)> generator) const {
        KernelSource src = to_source();
        src.msl_generator = std::move(generator);
        return src;
    }

    KernelSource to_source_with_spirv(std::vector<uint32_t> spirv_binary) const {
        KernelSource src = to_source();
        src.spirv_binary = std::move(spirv_binary);
        return src;
    }

    KernelSource to_source_with_spirv_generator(
        std::function<std::vector<uint32_t>(mlir::ModuleOp)> generator) const {
        KernelSource src = to_source();
        src.spirv_generator = std::move(generator);
        return src;
    }

    static KernelDispatch make_default_dispatch(const ov::Shape& shape,
                                                const ICompiledKernel& kernel) {
        return gfx_plugin::make_default_dispatch(shape, kernel.clamp_threadgroup_size(1));
    }

private:
    mlir::ModuleOp m_module;
    std::string m_entry_point;
    uint32_t m_arg_count = 0;
};

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

inline uint32_t infer_kernel_arg_count(mlir::ModuleOp module, const std::string& entry_point) {
    return infer_kernel_signature(module, entry_point).total();
}

}  // namespace gfx_plugin
}  // namespace ov
