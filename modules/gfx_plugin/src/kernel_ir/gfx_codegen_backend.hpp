// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "runtime/gpu_backend_base.hpp"

namespace ov {
namespace gfx_plugin {

struct KernelConstTensorSource {
    uint32_t value_id = 0;
    std::vector<uint8_t> bytes;
};

struct KernelSource {
    mlir::ModuleOp module;
    std::string entry_point;
    std::string msl_source;
    std::function<std::string(mlir::ModuleOp)> msl_generator;
    KernelSignature signature{};
    std::vector<KernelConstTensorSource> const_tensor_sources;
};

inline std::string resolve_entry_point(const KernelSource& source,
                                       std::string_view fallback = "gfx_kernel") {
    if (!source.entry_point.empty()) {
        return source.entry_point;
    }
    return std::string(fallback);
}

inline std::string resolve_msl_source(const KernelSource& source, std::string* log = nullptr) {
    std::string msl = source.msl_source;
    if (msl.empty() && source.msl_generator) {
        msl = source.msl_generator(source.module);
        if (msl.empty() && log) {
            *log = "MSL generator returned empty output";
        }
    }
    return msl;
}

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
