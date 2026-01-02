// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"
#include "runtime/gpu_backend_base.hpp"
#include "runtime/gpu_buffer_manager.hpp"
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

inline std::vector<KernelArg> materialize_kernel_bytes_args(const std::vector<KernelArg>& args,
                                                            GpuBufferManager& buffer_manager,
                                                            const char* stage_name) {
    bool has_bytes = false;
    for (const auto& arg : args) {
        if (arg.kind == KernelArg::Kind::Bytes) {
            has_bytes = true;
            break;
        }
    }
    if (!has_bytes) {
        return args;
    }
    OPENVINO_ASSERT(buffer_manager.supports_const_cache(),
                    "GFX: const cache is required for bytes args in stage ",
                    stage_name ? stage_name : "<unknown>");
    std::vector<KernelArg> out;
    out.reserve(args.size());
    for (const auto& arg : args) {
        if (arg.kind != KernelArg::Kind::Bytes) {
            out.push_back(arg);
            continue;
        }
        OPENVINO_ASSERT(arg.bytes,
                        "GFX: bytes arg pointer is null for stage ",
                        stage_name ? stage_name : "<unknown>");
        OPENVINO_ASSERT(arg.byte_size > 0,
                        "GFX: bytes arg size is zero for stage ",
                        stage_name ? stage_name : "<unknown>");
        const uint64_t data_hash = gfx_hash_bytes(arg.bytes, arg.byte_size);
        std::string key;
        key.reserve(64);
        key.append(stage_name ? stage_name : "GFX");
        key.append("/arg/");
        key.append(std::to_string(arg.index));
        key.append("/bytes/");
        key.append(std::to_string(arg.byte_size));
        key.append("/h/");
        key.append(std::to_string(data_hash));
        GpuBuffer buf = buffer_manager.wrap_const(key, arg.bytes, arg.byte_size, ov::element::u8);
        OPENVINO_ASSERT(buf.valid(),
                        "GFX: failed to materialize bytes arg for stage ",
                        stage_name ? stage_name : "<unknown>");
        out.push_back(make_buffer_arg(arg.index, buf, arg.offset));
    }
    return out;
}

class KernelArgsBuilder {
public:
    explicit KernelArgsBuilder(const char* stage_name)
        : m_stage_name(stage_name ? stage_name : "<unknown>") {}

    template <typename ResolveInputFn>
    void add_inputs(const std::vector<size_t>& kernel_inputs,
                    ResolveInputFn&& resolve_input) {
        OPENVINO_ASSERT(m_phase == Phase::Inputs && m_args.empty(),
                        "GFX: inputs must be added first for stage ",
                        m_stage_name);
        append_kernel_input_args(m_args,
                                 kernel_inputs,
                                 std::forward<ResolveInputFn>(resolve_input),
                                 m_stage_name);
    }

    template <typename ResolveInputFn>
    void add_inputs(size_t input_count,
                    ResolveInputFn&& resolve_input) {
        OPENVINO_ASSERT(m_phase == Phase::Inputs && m_args.empty(),
                        "GFX: inputs must be added first for stage ",
                        m_stage_name);
        append_kernel_input_args(m_args,
                                 input_count,
                                 std::forward<ResolveInputFn>(resolve_input),
                                 m_stage_name);
    }

    void add_input_buffer(const GpuBuffer& buffer, const char* label = nullptr) {
        OPENVINO_ASSERT(m_phase == Phase::Inputs,
                        "GFX: input buffers must be added before outputs for stage ",
                        m_stage_name);
        append_kernel_buffer_arg(m_args,
                                 next_index(),
                                 buffer,
                                 m_stage_name,
                                 label);
    }

    void add_optional_input_buffer(const GpuBuffer& buffer) {
        OPENVINO_ASSERT(m_phase == Phase::Inputs,
                        "GFX: optional input buffers must be added before outputs for stage ",
                        m_stage_name);
        append_kernel_optional_buffer_arg(m_args, next_index(), buffer);
    }

    void add_outputs(const std::vector<GpuTensor*>& outputs) {
        OPENVINO_ASSERT(m_phase != Phase::Params,
                        "GFX: outputs must be added before params for stage ",
                        m_stage_name);
        m_phase = Phase::Outputs;
        append_kernel_output_args(m_args,
                                  next_index(),
                                  outputs,
                                  m_stage_name);
    }

    void add_output(GpuTensor* output) {
        OPENVINO_ASSERT(m_phase != Phase::Params,
                        "GFX: output must be added before params for stage ",
                        m_stage_name);
        m_phase = Phase::Outputs;
        append_kernel_output_args(m_args,
                                  next_index(),
                                  output,
                                  m_stage_name);
    }

    void add_buffer(const GpuBuffer& buffer, const char* label = nullptr) {
        OPENVINO_ASSERT(m_phase != Phase::Inputs,
                        "GFX: params must be added after outputs for stage ",
                        m_stage_name);
        m_phase = Phase::Params;
        append_kernel_buffer_arg(m_args,
                                 next_index(),
                                 buffer,
                                 m_stage_name,
                                 label);
    }

    void add_optional_buffer(const GpuBuffer& buffer) {
        OPENVINO_ASSERT(m_phase != Phase::Inputs,
                        "GFX: params must be added after outputs for stage ",
                        m_stage_name);
        m_phase = Phase::Params;
        append_kernel_optional_buffer_arg(m_args, next_index(), buffer);
    }

    void add_bytes(const void* data, size_t size) {
        OPENVINO_ASSERT(m_phase != Phase::Inputs,
                        "GFX: params must be added after outputs for stage ",
                        m_stage_name);
        m_phase = Phase::Params;
        m_args.push_back(make_bytes_arg(next_index(), data, size));
    }

    const std::vector<KernelArg>& args() const { return m_args; }

    std::vector<KernelArg> finalize(GpuBufferManager* buffer_manager,
                                    const ICompiledKernel* kernel) const {
        std::vector<KernelArg> out = m_args;
        if (buffer_manager) {
            out = materialize_kernel_bytes_args(out, *buffer_manager, m_stage_name);
        }
        if (kernel) {
            validate_kernel_args(*kernel, out, m_stage_name);
        }
        return out;
    }

private:
    enum class Phase { Inputs, Outputs, Params };

    uint32_t next_index() const { return static_cast<uint32_t>(m_args.size()); }

    const char* m_stage_name = "<unknown>";
    Phase m_phase = Phase::Inputs;
    std::vector<KernelArg> m_args;
};

}  // namespace gfx_plugin
}  // namespace ov
