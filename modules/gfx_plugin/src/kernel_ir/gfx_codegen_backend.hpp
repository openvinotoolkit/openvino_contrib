// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "runtime/gfx_mpsrt_abi.hpp"
#include "runtime/gpu_backend_base.hpp"

namespace ov {
namespace gfx_plugin {

struct MpsrtConstTensorSource {
    GfxMpsrtValue value = 0;
    std::vector<uint8_t> bytes;
};

struct KernelSource {
    mlir::ModuleOp module;
    std::string entry_point;
    std::string msl_source;
    std::function<std::string(mlir::ModuleOp)> msl_generator;
    KernelSignature signature{};
    std::vector<MpsrtConstTensorSource> mpsrt_const_tensors;
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

inline uint32_t infer_msl_buffer_arg_count_from_source(std::string_view source) {
    uint32_t max_buffer_index = 0;
    bool found_buffer = false;
    size_t pos = 0;
    constexpr std::string_view marker = "[[buffer(";
    while ((pos = source.find(marker, pos)) != std::string_view::npos) {
        pos += marker.size();
        uint32_t index = 0;
        bool parsed_digit = false;
        while (pos < source.size() && source[pos] >= '0' && source[pos] <= '9') {
            parsed_digit = true;
            index = index * 10u + static_cast<uint32_t>(source[pos] - '0');
            ++pos;
        }
        if (parsed_digit) {
            found_buffer = true;
            max_buffer_index = std::max(max_buffer_index, index);
        }
    }
    return found_buffer ? max_buffer_index + 1u : 0u;
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
