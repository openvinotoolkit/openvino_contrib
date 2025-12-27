// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "runtime/gpu_backend.hpp"

namespace ov {
namespace gfx_plugin {

struct KernelSource {
    mlir::ModuleOp module;
    std::string entry_point;
    std::string msl_source;
    std::function<std::string(mlir::ModuleOp)> msl_generator;
    std::vector<uint32_t> spirv_binary;
    std::function<std::vector<uint32_t>(mlir::ModuleOp)> spirv_generator;
    KernelSignature signature{};
};

inline KernelSource make_kernel_source(mlir::ModuleOp module,
                                       std::string entry_point,
                                       std::string msl_source,
                                       uint32_t arg_count = 0) {
    KernelSource src;
    src.module = module;
    src.entry_point = std::move(entry_point);
    src.msl_source = std::move(msl_source);
    src.signature.arg_count = arg_count;
    return src;
}

inline KernelSource make_kernel_source_from_mlir(mlir::ModuleOp module,
                                                 std::string entry_point,
                                                 uint32_t arg_count = 0) {
    KernelSource src;
    src.module = module;
    src.entry_point = std::move(entry_point);
    src.signature.arg_count = arg_count;
    return src;
}

class ICodegenBackend {
public:
    virtual ~ICodegenBackend() = default;
    virtual std::shared_ptr<ICompiledKernel> compile(const KernelSource& source,
                                                     std::string* log = nullptr) = 0;
};

}  // namespace gfx_plugin
}  // namespace ov
