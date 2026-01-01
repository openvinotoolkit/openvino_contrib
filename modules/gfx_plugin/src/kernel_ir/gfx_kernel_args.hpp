// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <vector>

#include "openvino/core/except.hpp"
#include "runtime/gpu_backend_base.hpp"
#include "runtime/gpu_tensor.hpp"

namespace ov {
namespace gfx_plugin {

template <typename ResolveInputFn>
inline void append_kernel_input_args(std::vector<KernelArg>& args,
                                     const std::vector<size_t>& kernel_inputs,
                                     ResolveInputFn&& resolve_input,
                                     const char* stage_name) {
    for (size_t ai = 0; ai < kernel_inputs.size(); ++ai) {
        const size_t input_idx = kernel_inputs[ai];
        GpuTensor* t = resolve_input(input_idx);
        OPENVINO_ASSERT(t && t->buf.valid(),
                        "GFX: missing input buffer for stage ",
                        stage_name ? stage_name : "<unknown>");
        args.push_back(make_buffer_arg(static_cast<uint32_t>(ai), t->buf));
    }
}

template <typename ResolveInputFn>
inline void append_kernel_input_args(std::vector<KernelArg>& args,
                                     size_t input_count,
                                     ResolveInputFn&& resolve_input,
                                     const char* stage_name) {
    for (size_t ai = 0; ai < input_count; ++ai) {
        GpuTensor* t = resolve_input(ai);
        OPENVINO_ASSERT(t && t->buf.valid(),
                        "GFX: missing input buffer for stage ",
                        stage_name ? stage_name : "<unknown>");
        args.push_back(make_buffer_arg(static_cast<uint32_t>(ai), t->buf));
    }
}

inline void append_kernel_output_args(std::vector<KernelArg>& args,
                                      uint32_t base_index,
                                      const std::vector<GpuTensor*>& outputs,
                                      const char* stage_name) {
    for (size_t oi = 0; oi < outputs.size(); ++oi) {
        auto* out = outputs[oi];
        OPENVINO_ASSERT(out && out->buf.valid(),
                        "GFX: missing output buffer for stage ",
                        stage_name ? stage_name : "<unknown>");
        args.push_back(make_buffer_arg(base_index + static_cast<uint32_t>(oi), out->buf));
    }
}

inline void append_kernel_output_args(std::vector<KernelArg>& args,
                                      uint32_t base_index,
                                      GpuTensor* output,
                                      const char* stage_name) {
    OPENVINO_ASSERT(output && output->buf.valid(),
                    "GFX: missing output buffer for stage ",
                    stage_name ? stage_name : "<unknown>");
    args.push_back(make_buffer_arg(base_index, output->buf));
}

inline void append_kernel_buffer_arg(std::vector<KernelArg>& args,
                                     uint32_t index,
                                     const GpuBuffer& buffer,
                                     const char* stage_name,
                                     const char* label = nullptr) {
    OPENVINO_ASSERT(buffer.valid(),
                    "GFX: missing ",
                    label ? label : "buffer",
                    " for stage ",
                    stage_name ? stage_name : "<unknown>");
    args.push_back(make_buffer_arg(index, buffer));
}

inline void append_kernel_optional_buffer_arg(std::vector<KernelArg>& args,
                                              uint32_t index,
                                              const GpuBuffer& buffer) {
    args.push_back(make_buffer_arg(index, buffer));
}

inline uint32_t validate_kernel_args(const ICompiledKernel& kernel,
                                     const std::vector<KernelArg>& args,
                                     const char* stage_name) {
    const uint32_t count = ensure_kernel_args_dense(args, stage_name);
    const uint32_t expected = kernel.args_count();
    if (expected) {
        OPENVINO_ASSERT(expected == count,
                        stage_name ? stage_name : "GFX",
                        ": kernel args count mismatch (expected ",
                        expected,
                        ", got ",
                        count,
                        ")");
    }
    return count;
}

}  // namespace gfx_plugin
}  // namespace ov
