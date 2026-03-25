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
        // Keep bytes-arg materialization on a backend-neutral immutable namespace so
        // equal payloads across different stages can reuse the same device buffer.
        static const std::string key = "kernel-bytes-args";
        GpuBuffer buf = buffer_manager.wrap_const(key, arg.bytes, arg.byte_size, ov::element::u8);
        OPENVINO_ASSERT(buf.valid(),
                        "GFX: failed to materialize bytes arg for stage ",
                        stage_name ? stage_name : "<unknown>");
        out.push_back(make_buffer_arg(arg.index, buf, arg.offset));
    }
    return out;
}

struct KernelArgsBundle {
    std::vector<KernelArg> args;
    std::vector<int32_t> scalar_storage;
};

inline void append_kernel_arg_debug(std::string* log, size_t op_idx, const std::string& value) {
    if (!log) {
        return;
    }
    if (log->empty()) {
        log->append("Kernel arg map: ");
    } else {
        log->append(", ");
    }
    log->append("arg");
    log->append(std::to_string(op_idx));
    log->append("=");
    log->append(value);
}

template <typename ResolveInputFn>
inline KernelArgsBundle build_kernel_args_from_metadata(const std::vector<int32_t>& operand_kinds,
                                                        const std::vector<int32_t>& operand_arg_indices,
                                                        const std::vector<int32_t>& scalar_args,
                                                        const std::vector<size_t>& kernel_inputs,
                                                        size_t kernel_input_arg_count,
                                                        const std::vector<GpuTensor>& extra_inputs,
                                                        const std::vector<GpuTensor*>& outputs,
                                                        ResolveInputFn&& resolve_input,
                                                        const char* stage_name,
                                                        std::string* debug_log = nullptr) {
    KernelArgsBundle bundle;
    const char* label = stage_name ? stage_name : "<unknown>";
    uint32_t arg_index = 0;

    if (operand_kinds.empty()) {
        bundle.args.reserve(scalar_args.size() + kernel_inputs.size() + extra_inputs.size() + outputs.size());
        bundle.scalar_storage = scalar_args;
        for (auto& v : bundle.scalar_storage) {
            bundle.args.push_back(make_bytes_arg(arg_index++, &v, sizeof(v)));
        }
        for (size_t ai = 0; ai < kernel_inputs.size(); ++ai) {
            const size_t input_idx = kernel_inputs[ai];
            GpuTensor* t = resolve_input(input_idx);
            OPENVINO_ASSERT(t && t->buf.valid(),
                            "GFX: missing input buffer for stage ",
                            label);
            bundle.args.push_back(make_buffer_arg(arg_index++, t->buf));
        }
        for (const auto& extra : extra_inputs) {
            OPENVINO_ASSERT(extra.buf.valid(),
                            "GFX: missing extra buffer for stage ",
                            label);
            bundle.args.push_back(make_buffer_arg(arg_index++, extra.buf));
        }
        for (auto* out : outputs) {
            OPENVINO_ASSERT(out && out->buf.valid(),
                            "GFX: missing output buffer for stage ",
                            label);
            bundle.args.push_back(make_buffer_arg(arg_index++, out->buf));
        }
        return bundle;
    }

    bundle.args.reserve(operand_kinds.size());
    size_t scalar_count = 0;
    for (auto kind : operand_kinds) {
        if (kind == 0) {
            ++scalar_count;
        }
    }
    bundle.scalar_storage.reserve(scalar_count);

    size_t scalar_idx = 0;
    size_t input_pos = 0;
    size_t output_pos = 0;
    size_t extra_pos = 0;
    const bool has_arg_indices = operand_arg_indices.size() == operand_kinds.size();
    const size_t input_arg_count = (kernel_input_arg_count != 0) ? kernel_input_arg_count : kernel_inputs.size();

    for (size_t op_idx = 0; op_idx < operand_kinds.size(); ++op_idx) {
        const auto kind = operand_kinds[op_idx];
        if (kind == 0) {
            int32_t value = 0;
            if (scalar_idx < scalar_args.size()) {
                value = scalar_args[scalar_idx++];
            }
            bundle.scalar_storage.push_back(value);
            bundle.args.push_back(make_bytes_arg(arg_index++, &bundle.scalar_storage.back(), sizeof(value)));
            append_kernel_arg_debug(debug_log, op_idx, std::string("scalar(") + std::to_string(value) + ")");
            continue;
        }

        int32_t arg_idx = -1;
        if (has_arg_indices) {
            arg_idx = operand_arg_indices[op_idx];
        }
        if (arg_idx >= 0) {
            const size_t uarg = static_cast<size_t>(arg_idx);
            if (uarg < input_arg_count) {
                if (uarg < kernel_inputs.size()) {
                    const size_t input_idx = kernel_inputs[uarg];
                    GpuTensor* t = resolve_input(input_idx);
                    OPENVINO_ASSERT(t && t->buf.valid(),
                                    "GFX: missing input buffer for stage ",
                                    label);
                    bundle.args.push_back(make_buffer_arg(arg_index++, t->buf));
                    append_kernel_arg_debug(debug_log,
                                            op_idx,
                                            std::string("input[") + std::to_string(input_idx) + "]");
                    continue;
                }
                const size_t extra_idx = uarg - kernel_inputs.size();
                OPENVINO_ASSERT(extra_idx < extra_inputs.size(),
                                "GFX: missing extra buffer for stage ",
                                label);
                const auto& extra = extra_inputs[extra_idx];
                OPENVINO_ASSERT(extra.buf.valid(),
                                "GFX: missing extra buffer for stage ",
                                label);
                bundle.args.push_back(make_buffer_arg(arg_index++, extra.buf));
                append_kernel_arg_debug(debug_log,
                                        op_idx,
                                        std::string("extra[") + std::to_string(extra_idx) + "]");
                continue;
            }
            const size_t out_idx = uarg - input_arg_count;
            if (out_idx < outputs.size()) {
                auto* out = outputs[out_idx];
                OPENVINO_ASSERT(out && out->buf.valid(),
                                "GFX: missing output buffer for stage ",
                                label);
                bundle.args.push_back(make_buffer_arg(arg_index++, out->buf));
                append_kernel_arg_debug(debug_log,
                                        op_idx,
                                        std::string("output[") + std::to_string(out_idx) + "]");
                continue;
            }
        }

        if (extra_pos < extra_inputs.size()) {
            const auto& extra = extra_inputs[extra_pos++];
            OPENVINO_ASSERT(extra.buf.valid(),
                            "GFX: missing extra buffer for stage ",
                            label);
            bundle.args.push_back(make_buffer_arg(arg_index++, extra.buf));
            append_kernel_arg_debug(debug_log,
                                    op_idx,
                                    std::string("extra[") + std::to_string(extra_pos - 1) + "]");
            continue;
        }
        if (input_pos < kernel_inputs.size()) {
            const size_t input_idx = kernel_inputs[input_pos++];
            GpuTensor* t = resolve_input(input_idx);
            OPENVINO_ASSERT(t && t->buf.valid(),
                            "GFX: missing input buffer for stage ",
                            label);
            bundle.args.push_back(make_buffer_arg(arg_index++, t->buf));
            append_kernel_arg_debug(debug_log,
                                    op_idx,
                                    std::string("input[") + std::to_string(input_idx) + "]");
            continue;
        }
        OPENVINO_ASSERT(output_pos < outputs.size(),
                        "GFX: missing output buffer for stage ",
                        label);
        auto* out = outputs[output_pos++];
        OPENVINO_ASSERT(out && out->buf.valid(),
                        "GFX: missing output buffer for stage ",
                        label);
        bundle.args.push_back(make_buffer_arg(arg_index++, out->buf));
        append_kernel_arg_debug(debug_log,
                                op_idx,
                                std::string("output[") + std::to_string(output_pos - 1) + "]");
    }

    return bundle;
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
