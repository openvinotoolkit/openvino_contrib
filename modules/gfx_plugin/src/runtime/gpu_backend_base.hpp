// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "openvino/core/except.hpp"
#include "runtime/gpu_types.hpp"

namespace ov {
namespace gfx_plugin {

struct KernelArg {
    enum class Kind { Buffer, Bytes };
    Kind kind = Kind::Buffer;
    uint32_t index = 0;
    size_t offset = 0;
    GpuBuffer buffer{};
    const void* bytes = nullptr;
    size_t byte_size = 0;
};

struct KernelSignature {
    uint32_t arg_count = 0;
};

inline KernelArg make_buffer_arg(uint32_t index, const GpuBuffer& buffer, size_t offset = 0) {
    KernelArg arg;
    arg.kind = KernelArg::Kind::Buffer;
    arg.index = index;
    arg.offset = offset;
    arg.buffer = buffer;
    return arg;
}

inline KernelArg make_bytes_arg(uint32_t index, const void* data, size_t size) {
    KernelArg arg;
    arg.kind = KernelArg::Kind::Bytes;
    arg.index = index;
    arg.bytes = data;
    arg.byte_size = size;
    return arg;
}

struct KernelDispatch {
    size_t grid[3] = {1, 1, 1};
    size_t threads_per_group[3] = {1, 1, 1};
};

struct KernelExecutionHooks {
    std::function<void(GpuCommandEncoderHandle)> on_begin;
    std::function<void(GpuCommandEncoderHandle)> on_end;
    std::function<void()> on_complete;
};

inline uint32_t kernel_args_count(const std::vector<KernelArg>& args) {
    uint32_t max_index = 0;
    bool found = false;
    for (const auto& arg : args) {
        if (arg.kind != KernelArg::Kind::Buffer && arg.kind != KernelArg::Kind::Bytes) {
            continue;
        }
        max_index = std::max(max_index, arg.index);
        found = true;
    }
    return found ? (max_index + 1u) : 0u;
}

inline bool kernel_args_dense(const std::vector<KernelArg>& args, uint32_t* out_count = nullptr) {
    const uint32_t count = kernel_args_count(args);
    if (out_count) {
        *out_count = count;
    }
    if (count == 0) {
        return true;
    }
    std::vector<bool> seen(count, false);
    for (const auto& arg : args) {
        if (arg.kind != KernelArg::Kind::Buffer && arg.kind != KernelArg::Kind::Bytes) {
            continue;
        }
        if (arg.index >= count) {
            return false;
        }
        if (seen[arg.index]) {
            return false;
        }
        seen[arg.index] = true;
    }
    return std::all_of(seen.begin(), seen.end(), [](bool v) { return v; });
}

inline uint32_t ensure_kernel_args_dense(const std::vector<KernelArg>& args, const char* label) {
    uint32_t count = 0;
    OPENVINO_ASSERT(kernel_args_dense(args, &count),
                    label ? label : "GFX",
                    ": kernel args must be densely indexed from 0");
    return count;
}

class ICompiledKernel {
public:
    virtual ~ICompiledKernel() = default;
    virtual uint32_t args_count() const { return 0; }
    virtual void set_args_count(uint32_t /*count*/) {}
    virtual size_t clamp_threadgroup_size(size_t desired) const = 0;
    virtual void execute(GpuCommandBufferHandle command_buffer,
                         const KernelDispatch& dispatch,
                         const std::vector<KernelArg>& args,
                         const KernelExecutionHooks* hooks = nullptr) = 0;
};

}  // namespace gfx_plugin
}  // namespace ov
