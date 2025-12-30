// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "openvino/core/shape.hpp"
#include "mlir/gfx_codegen_backend.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "runtime/gfx_kernel_dispatch.hpp"

namespace ov {
namespace gfx_plugin {

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
        return make_kernel_source_from_mlir(m_module, m_entry_point, m_arg_count);
    }

    KernelSource to_source_with_msl(std::string msl_source) const {
        KernelSource src = to_source();
        src.msl_source = std::move(msl_source);
        return src;
    }

    KernelSource to_source_with_spirv(std::vector<uint32_t> spirv_binary) const {
        KernelSource src = to_source();
        src.spirv_binary = std::move(spirv_binary);
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

inline uint32_t infer_kernel_arg_count(mlir::ModuleOp module, const std::string& entry_point) {
    if (!module) {
        return 0;
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
        return 0;
    }
    auto ftype = func.getFunctionType();
    return static_cast<uint32_t>(ftype.getNumInputs() + ftype.getNumResults());
}

}  // namespace gfx_plugin
}  // namespace ov
