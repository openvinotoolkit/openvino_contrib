// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/runtime/opencl_source_stage.hpp"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "backends/opencl/runtime/opencl_program_cache.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "mlir/gfx_stage_runtime_values.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "runtime/gpu_backend_base.hpp"
#include "runtime/gfx_logger.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

uint32_t checked_element_count(const ov::Shape& shape, const char* label) {
    const auto elements = ov::shape_size(shape);
    OPENVINO_ASSERT(elements <= std::numeric_limits<uint32_t>::max(),
                    label,
                    ": OpenCL baseline kernel supports at most uint32 element counts");
    return static_cast<uint32_t>(elements);
}

uint32_t scalar_value_for_opencl_source_arg(GfxOpenClSourceScalarArg scalar,
                                            uint32_t element_count,
                                            GfxOpenClArtifactOp op,
                                            GfxOpenClArtifactInputMode input_mode,
                                            float scalar_constant_f32,
                                            const std::vector<ov::Shape>& input_shapes,
                                            const ov::Shape& output0_shape,
                                            const std::vector<uint32_t>& static_u32_scalars,
                                            const std::vector<float>& static_f32_scalars,
                                            size_t& static_u32_idx,
                                            size_t& static_f32_idx) {
    const auto raw_scalar = static_cast<uint32_t>(scalar);
    auto resolve_dim = [&](GfxOpenClSourceScalarArg base,
                           size_t shape_idx,
                           const std::vector<ov::Shape>& shapes) -> std::optional<uint32_t> {
        const auto base_value = static_cast<uint32_t>(base);
        if (raw_scalar < base_value || raw_scalar >= base_value + 8u) {
            return std::nullopt;
        }
        const size_t axis = static_cast<size_t>(raw_scalar - base_value);
        if (shape_idx >= shapes.size() || axis >= shapes[shape_idx].size()) {
            return 0;
        }
        OPENVINO_ASSERT(shapes[shape_idx][axis] <= std::numeric_limits<uint32_t>::max(),
                        "GFX OpenCL: runtime input dim exceeds source scalar ABI");
        return static_cast<uint32_t>(shapes[shape_idx][axis]);
    };
    for (size_t input_idx = 0; input_idx < 4; ++input_idx) {
        const auto base = static_cast<GfxOpenClSourceScalarArg>(
            static_cast<uint32_t>(GfxOpenClSourceScalarArg::Input0Dim0) +
            static_cast<uint32_t>(input_idx * 8));
        if (const auto value = resolve_dim(base, input_idx, input_shapes)) {
            return *value;
        }
    }
    if (raw_scalar >= static_cast<uint32_t>(GfxOpenClSourceScalarArg::Output0Dim0) &&
        raw_scalar <= static_cast<uint32_t>(GfxOpenClSourceScalarArg::Output0Dim7)) {
        const size_t axis = static_cast<size_t>(
            raw_scalar - static_cast<uint32_t>(GfxOpenClSourceScalarArg::Output0Dim0));
        if (axis >= output0_shape.size()) {
            return 0;
        }
        OPENVINO_ASSERT(output0_shape[axis] <= std::numeric_limits<uint32_t>::max(),
                        "GFX OpenCL: runtime output dim exceeds source scalar ABI");
        return static_cast<uint32_t>(output0_shape[axis]);
    }
    switch (scalar) {
        case GfxOpenClSourceScalarArg::ElementCount:
            return element_count;
        case GfxOpenClSourceScalarArg::OpCode:
            return static_cast<uint32_t>(op);
        case GfxOpenClSourceScalarArg::InputMode:
            return static_cast<uint32_t>(input_mode);
        case GfxOpenClSourceScalarArg::ScalarConstantF32: {
            uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(scalar_constant_f32),
                          "GFX OpenCL: f32 scalar ABI must be 32-bit");
            std::memcpy(&bits, &scalar_constant_f32, sizeof(bits));
            return bits;
        }
        case GfxOpenClSourceScalarArg::StaticU32:
            OPENVINO_ASSERT(static_u32_idx < static_u32_scalars.size(),
                            "GFX OpenCL: static u32 scalar ABI has no value mapping");
            return static_u32_scalars[static_u32_idx++];
        case GfxOpenClSourceScalarArg::StaticF32: {
            OPENVINO_ASSERT(static_f32_idx < static_f32_scalars.size(),
                            "GFX OpenCL: static f32 scalar ABI has no value mapping");
            const float value = static_f32_scalars[static_f32_idx++];
            uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value),
                          "GFX OpenCL: f32 scalar ABI must be 32-bit");
            std::memcpy(&bits, &value, sizeof(bits));
            return bits;
        }
        default:
            break;
    }
    OPENVINO_THROW("GFX OpenCL: unsupported source scalar argument kind");
}

size_t round_up(size_t value, size_t step) {
    if (step == 0) {
        return value;
    }
    return ((value + step - 1) / step) * step;
}

bool is_linear_shape_view_op(std::string_view type) {
    return type == "Reshape" || type == "Squeeze" || type == "Unsqueeze";
}

class OpenClSourceStage final : public GpuStage {
public:
    OpenClSourceStage(std::shared_ptr<const ov::Node> node,
                      std::shared_ptr<OpenClRuntimeContext> context,
                      RuntimeStageExecutableDescriptor descriptor,
                      GfxOpenClSourceArtifact artifact)
        : m_node(std::move(node)),
          m_context(std::move(context)),
          m_descriptor(std::move(descriptor)),
          m_artifact(std::move(artifact)) {
        OPENVINO_ASSERT(m_node, "GFX OpenCL: source stage requires a node");
        OPENVINO_ASSERT(m_context, "GFX OpenCL: source stage requires a runtime context");
        OPENVINO_ASSERT(m_descriptor.payload_kind == compiler::KernelArtifactPayloadKind::OpenClSource,
                        "GFX OpenCL: source stage requires OpenCL source runtime descriptor");
        OPENVINO_ASSERT(m_descriptor.backend_domain == "opencl",
                        "GFX OpenCL: source stage descriptor backend domain drift");
        OPENVINO_ASSERT(m_descriptor.entry_point == m_artifact.artifact_ref.entry_point,
                        "GFX OpenCL: source stage descriptor entry point drift");
        m_program_cache = std::make_shared<OpenClProgramCache>(m_context);
        m_name = m_node->get_friendly_name();
        m_type = m_node->get_type_name();
    }

    void init(GpuBufferManager* buffer_manager) override {
        m_buffer_manager = buffer_manager;
    }

    void compile(GpuBufferManager* buffer_manager) override {
        if (buffer_manager) {
            m_buffer_manager = buffer_manager;
        }
        OPENVINO_ASSERT(m_artifact.valid, "GFX OpenCL: invalid source artifact for ", m_type);
        if (m_kernel) {
            return;
        }
        if (should_execute_chunked_static_concat()) {
            prepare_planned_chunk_kernels(m_concat_chunk_kernels);
            return;
        }
        if (should_execute_chunked_static_split()) {
            prepare_planned_chunk_kernels(m_split_chunk_kernels);
            return;
        }
        m_kernel = m_program_cache->get_or_create(m_artifact.artifact_ref.source_id,
                                                  m_artifact.source,
                                                  m_artifact.artifact_ref.entry_point,
                                                  gfx_opencl_source_artifact_build_options(m_artifact));
        m_kernel->set_args_count(m_artifact.arg_count);
    }

    void execute(GpuCommandBufferHandle command_buffer) override {
        if (!m_kernel &&
            !should_execute_chunked_static_concat() &&
            !should_execute_chunked_static_split()) {
            compile(m_buffer_manager);
        }
        const auto outputs = resolve_outputs();
        OPENVINO_ASSERT(outputs.size() == m_artifact.direct_output_count,
                        "GFX OpenCL: output binding count does not match source artifact ABI for ",
                        m_name);
        OPENVINO_ASSERT(!outputs.empty(),
                        "GFX OpenCL: source artifact must bind at least one output for ",
                        m_name);
        if (try_alias_linear_shape_view(outputs) ||
            try_alias_linear_slice_view(outputs)) {
            return;
        }
        for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
            GpuTensor* output = outputs[output_idx];
            OPENVINO_ASSERT(output && output->buf.valid(),
                            "GFX OpenCL: output buffer ",
                            output_idx,
                            " is not materialized for ",
                            m_name);
            auto output_type = output->expected_type;
            if (output_type == ov::element::dynamic && m_node &&
                output_idx < m_node->get_output_size()) {
                output_type = m_node->get_output_element_type(output_idx);
            }
            OPENVINO_ASSERT(output_type == ov::element::dynamic ||
                                output_type == ov::element::f16 ||
                                output_type == ov::element::f32 ||
                                output_type == ov::element::boolean ||
                                output_type == ov::element::i32 ||
                                output_type == ov::element::i64,
                            "GFX OpenCL: baseline stage currently supports f32, boolean, i32 and i64 outputs only");
        }
        const auto count = checked_element_count(resolve_element_count_shape(outputs),
                                                 "GFX OpenCL");
        OPENVINO_ASSERT(count > 0, "GFX OpenCL: zero-sized baseline dispatch is not supported yet");

        if (try_execute_chunked_static_concat(command_buffer, outputs, count) ||
            try_execute_chunked_static_split(command_buffer, outputs, count)) {
            return;
        }

        const auto roles = materialize_gfx_kernel_external_buffer_roles(
            m_artifact.stage_manifest.custom_kernel.external_buffer_abi);
        OPENVINO_ASSERT(!roles.empty(),
                        "GFX OpenCL: source artifact is missing role-based ABI for ",
                        m_name);
        OPENVINO_ASSERT(m_artifact.arg_count == roles.size(),
                        "GFX OpenCL: source artifact arg count does not match role ABI for ",
                        m_name);
        OPENVINO_ASSERT(m_artifact.direct_input_indices.size() == m_artifact.direct_input_count,
                        "GFX OpenCL: source artifact direct input index metadata is inconsistent for ",
                        m_name);

        std::vector<KernelArg> args;
        args.reserve(roles.size());
        m_scalar_storage.clear();
        m_scalar_storage.reserve(m_artifact.scalar_args.size());
        std::vector<ov::Shape> input_shapes;
        input_shapes.reserve(std::min<size_t>(m_node ? m_node->get_input_size() : 0, 4));
        for (size_t idx = 0; m_node && idx < m_node->get_input_size() && idx < 4; ++idx) {
            input_shapes.push_back(resolve_input_shape(idx));
        }
        const ov::Shape output0_shape = resolve_output_shape(*outputs.front(), 0);
        size_t input_idx = 0;
        size_t output_idx = 0;
        size_t scalar_idx = 0;
        size_t static_u32_idx = 0;
        size_t static_f32_idx = 0;
        for (size_t arg_idx = 0; arg_idx < roles.size(); ++arg_idx) {
            switch (roles[arg_idx]) {
                case GfxKernelBufferRole::TensorInput: {
                    OPENVINO_ASSERT(input_idx < m_artifact.direct_input_indices.size(),
                                    "GFX OpenCL: tensor ABI has no input slot mapping for ",
                                    m_name);
                    const size_t node_input_idx = m_artifact.direct_input_indices[input_idx];
                    GpuTensor* input = resolve_tensor_input(node_input_idx);
                    OPENVINO_ASSERT(input && input->buf.valid(),
                                    "GFX OpenCL: input slot ",
                                    node_input_idx,
                                    " is not materialized for ",
                                    m_name);
                    args.push_back(make_buffer_arg(static_cast<uint32_t>(arg_idx),
                                                   input->buf));
                    ++input_idx;
                    break;
                }
                case GfxKernelBufferRole::TensorOutput: {
                    OPENVINO_ASSERT(output_idx < outputs.size(),
                                    "GFX OpenCL: tensor ABI has no output slot mapping for ",
                                    m_name);
                    args.push_back(make_buffer_arg(static_cast<uint32_t>(arg_idx),
                                                   outputs[output_idx]->buf));
                    ++output_idx;
                    break;
                }
                case GfxKernelBufferRole::ScalarParam: {
                    OPENVINO_ASSERT(scalar_idx < m_artifact.scalar_args.size(),
                                    "GFX OpenCL: scalar ABI has no value mapping for ",
                                    m_name);
                    m_scalar_storage.push_back(scalar_value_for_opencl_source_arg(
                        m_artifact.scalar_args[scalar_idx],
                        count,
                        m_artifact.op,
                        m_artifact.input_mode,
                        m_artifact.scalar_constant_f32,
                        input_shapes,
                        output0_shape,
                        m_artifact.static_u32_scalars,
                        m_artifact.static_f32_scalars,
                        static_u32_idx,
                        static_f32_idx));
                    args.push_back(make_bytes_arg(static_cast<uint32_t>(arg_idx),
                                                  &m_scalar_storage.back(),
                                                  sizeof(m_scalar_storage.back())));
                    ++scalar_idx;
                    break;
                }
                case GfxKernelBufferRole::ConstTensor:
                case GfxKernelBufferRole::RuntimeParams:
                case GfxKernelBufferRole::Unknown:
                default:
                    OPENVINO_THROW("GFX OpenCL: unsupported role in source artifact ABI for ",
                                   m_name);
            }
        }
        OPENVINO_ASSERT(input_idx == m_artifact.direct_input_count,
                        "GFX OpenCL: role ABI input count does not match artifact input count for ",
                        m_name);
        OPENVINO_ASSERT(output_idx == m_artifact.direct_output_count,
                        "GFX OpenCL: role ABI output count does not match artifact output count for ",
                        m_name);
        OPENVINO_ASSERT(scalar_idx == m_artifact.scalar_args.size(),
                        "GFX OpenCL: not all source scalar args were consumed for ",
                        m_name);
        OPENVINO_ASSERT(static_u32_idx == m_artifact.static_u32_scalars.size(),
                        "GFX OpenCL: not all source static u32 scalars were consumed for ",
                        m_name);
        OPENVINO_ASSERT(static_f32_idx == m_artifact.static_f32_scalars.size(),
                        "GFX OpenCL: not all source static f32 scalars were consumed for ",
                        m_name);

        if (gfx_log_debug_enabled()) {
            std::ostringstream oss;
            oss << "source_stage name=" << m_name
                << " entry=" << m_artifact.artifact_ref.entry_point
                << " count=" << count
                << " inputs=" << input_shapes.size()
                << " output0=[";
            for (size_t i = 0; i < output0_shape.size(); ++i) {
                if (i) {
                    oss << ",";
                }
                oss << output0_shape[i];
            }
            oss << "] scalars=[";
            for (size_t i = 0; i < m_scalar_storage.size(); ++i) {
                if (i) {
                    oss << ",";
                }
                oss << m_scalar_storage[i];
            }
            oss << "]";
            for (size_t input_idx = 0; input_idx < input_shapes.size(); ++input_idx) {
                oss << " input" << input_idx << "=[";
                const auto& shape = input_shapes[input_idx];
                for (size_t dim = 0; dim < shape.size(); ++dim) {
                    if (dim) {
                        oss << ",";
                    }
                    oss << shape[dim];
                }
                oss << "]";
            }
            gfx_log_debug("OpenCLSource") << oss.str();
        }

        const size_t local = std::max<size_t>(1, m_kernel->clamp_threadgroup_size(m_artifact.local_size_hint));
        KernelDispatch dispatch{};
        dispatch.grid[0] = round_up(count, local);
        dispatch.grid[1] = 1;
        dispatch.grid[2] = 1;
        dispatch.threads_per_group[0] = local;
        dispatch.threads_per_group[1] = 1;
        dispatch.threads_per_group[2] = 1;
        m_kernel->execute(command_buffer, dispatch, args);
    }

    void set_inputs(const std::vector<GpuTensor*>& inputs) override {
        m_inputs = inputs;
    }

    void set_output(GpuTensor* output) override {
        m_output = output;
        m_outputs.clear();
        if (output) {
            m_outputs.push_back(output);
        }
    }

    void set_output_refs(const std::vector<GpuTensor*>& outputs) override {
        m_outputs = outputs;
        m_output = outputs.empty() ? nullptr : outputs.front();
    }

    const std::string& name() const override { return m_name; }
    const std::string& type() const override { return m_type; }

    std::unique_ptr<GpuStage> clone() const override {
        auto cloned = std::make_unique<OpenClSourceStage>(m_node,
                                                          m_context,
                                                          m_descriptor,
                                                          m_artifact);
        cloned->m_name = m_name;
        cloned->m_type = m_type;
        cloned->m_program_cache = m_program_cache;
        if (m_kernel) {
            cloned->m_kernel = m_kernel->fork();
        }
        cloned->m_concat_chunk_kernels.reserve(m_concat_chunk_kernels.size());
        for (const auto& kernel : m_concat_chunk_kernels) {
            cloned->m_concat_chunk_kernels.push_back(kernel ? kernel->fork() : nullptr);
        }
        cloned->m_split_chunk_kernels.reserve(m_split_chunk_kernels.size());
        for (const auto& kernel : m_split_chunk_kernels) {
            cloned->m_split_chunk_kernels.push_back(kernel ? kernel->fork() : nullptr);
        }
        return cloned;
    }

private:
    bool should_execute_chunked_static_concat() const {
        if (m_artifact.input_chunk_size == 0 ||
            m_artifact.stage_manifest.stage_family != GfxKernelStageFamily::ConcatSplit ||
            m_artifact.direct_input_count <= 4 ||
            m_artifact.direct_output_count != 1 ||
            m_artifact.direct_input_indices.size() != m_artifact.direct_input_count ||
            m_artifact.source_static_u32_scalars.size() !=
                2 + static_cast<size_t>(m_artifact.direct_input_count) * 2 ||
            m_artifact.planned_chunks.empty()) {
            return false;
        }
        uint32_t next_input_begin = 0;
        for (const auto& chunk : m_artifact.planned_chunks) {
            if (chunk.binding_begin != next_input_begin ||
                chunk.binding_count == 0 ||
                chunk.binding_count > m_artifact.input_chunk_size ||
                !chunk.artifact ||
                !chunk.artifact->valid ||
                chunk.artifact->direct_input_count != chunk.binding_count ||
                chunk.artifact->direct_output_count != 1 ||
                chunk.artifact->direct_input_indices.size() != chunk.binding_count ||
                chunk.artifact->source_static_u32_scalars.size() !=
                    2 + static_cast<size_t>(chunk.binding_count) * 2) {
                return false;
            }
            next_input_begin += chunk.binding_count;
        }
        if (next_input_begin != m_artifact.direct_input_count) {
            return false;
        }
        return true;
    }

    bool try_execute_chunked_static_concat(GpuCommandBufferHandle command_buffer,
                                           const std::vector<GpuTensor*>& outputs,
                                           uint32_t count) {
        if (!should_execute_chunked_static_concat()) {
            return false;
        }
        OPENVINO_ASSERT(outputs.size() == 1 && outputs.front() &&
                            outputs.front()->buf.valid(),
                        "GFX OpenCL: chunked Concat output is not materialized for ",
                        m_name);
        GpuTensor* output = outputs.front();

        for (size_t chunk_slot = 0; chunk_slot < m_artifact.planned_chunks.size();
             ++chunk_slot) {
            const auto& planned_chunk = m_artifact.planned_chunks[chunk_slot];
            const auto& chunk_artifact = *planned_chunk.artifact;
            const uint32_t chunk_inputs = planned_chunk.binding_count;
            const uint32_t axis_total = chunk_artifact.source_static_u32_scalars[0];
            OPENVINO_ASSERT(axis_total > 0 && count % axis_total == 0,
                            "GFX OpenCL: chunked Concat has invalid axis/count metadata for ",
                            m_name);
            uint64_t chunk_axis_total = 0;
            for (uint32_t local_input = 0; local_input < chunk_inputs; ++local_input) {
                const size_t len_idx =
                    2 + static_cast<size_t>(local_input) * 2 + 1;
                chunk_axis_total += chunk_artifact.source_static_u32_scalars[len_idx];
            }
            const uint64_t chunk_count64 = (static_cast<uint64_t>(count) / axis_total) *
                                           chunk_axis_total;
            OPENVINO_ASSERT(chunk_count64 > 0 &&
                                chunk_count64 <= std::numeric_limits<uint32_t>::max(),
                            "GFX OpenCL: chunked Concat element count exceeds OpenCL scalar range for ",
                            m_name);
            const uint32_t chunk_count = static_cast<uint32_t>(chunk_count64);

            auto& kernel =
                prepare_planned_chunk_kernel(m_concat_chunk_kernels, chunk_slot, chunk_artifact);

            std::vector<KernelArg> args;
            args.reserve(2 + chunk_inputs);
            for (uint32_t local_input = 0; local_input < chunk_inputs; ++local_input) {
                const size_t node_input_idx =
                    chunk_artifact.direct_input_indices[local_input];
                GpuTensor* input = resolve_tensor_input(node_input_idx);
                OPENVINO_ASSERT(input && input->buf.valid(),
                                "GFX OpenCL: chunked Concat input ",
                                node_input_idx,
                                " is not materialized for ",
                                m_name);
                args.push_back(make_buffer_arg(local_input, input->buf));
            }
            args.push_back(make_buffer_arg(chunk_inputs, output->buf));
            uint32_t count_scalar = chunk_count;
            args.push_back(make_bytes_arg(chunk_inputs + 1,
                                          &count_scalar,
                                          sizeof(count_scalar)));

            const size_t local = std::max<size_t>(
                1,
                kernel->clamp_threadgroup_size(chunk_artifact.local_size_hint));
            KernelDispatch dispatch{};
            dispatch.grid[0] = round_up(chunk_count, local);
            dispatch.grid[1] = 1;
            dispatch.grid[2] = 1;
            dispatch.threads_per_group[0] = local;
            dispatch.threads_per_group[1] = 1;
            dispatch.threads_per_group[2] = 1;
            kernel->execute(command_buffer, dispatch, args);
        }
        return true;
    }

    bool should_execute_chunked_static_split() const {
        if (m_artifact.output_chunk_size == 0 ||
            m_artifact.stage_manifest.stage_family != GfxKernelStageFamily::ConcatSplit ||
            m_artifact.direct_output_count <= 4 ||
            m_artifact.direct_input_count != 1 ||
            m_artifact.source_static_u32_scalars.size() !=
                2 + static_cast<size_t>(m_artifact.direct_output_count) * 2 ||
            m_artifact.planned_chunks.empty()) {
            return false;
        }
        uint32_t next_output_begin = 0;
        for (const auto& chunk : m_artifact.planned_chunks) {
            if (chunk.binding_begin != next_output_begin ||
                chunk.binding_count == 0 ||
                chunk.binding_count > m_artifact.output_chunk_size ||
                !chunk.artifact ||
                !chunk.artifact->valid ||
                chunk.artifact->direct_input_count != 1 ||
                chunk.artifact->direct_output_count != chunk.binding_count ||
                chunk.artifact->source_static_u32_scalars.size() !=
                    2 + static_cast<size_t>(chunk.binding_count) * 2) {
                return false;
            }
            next_output_begin += chunk.binding_count;
        }
        if (next_output_begin != m_artifact.direct_output_count) {
            return false;
        }
        return true;
    }

    bool try_execute_chunked_static_split(GpuCommandBufferHandle command_buffer,
                                          const std::vector<GpuTensor*>& outputs,
                                          uint32_t count) {
        if (!should_execute_chunked_static_split()) {
            return false;
        }
        OPENVINO_ASSERT(!m_artifact.direct_input_indices.empty(),
                        "GFX OpenCL: chunked Split artifact has no input slot");
        GpuTensor* input = resolve_tensor_input(m_artifact.direct_input_indices.front());
        OPENVINO_ASSERT(input && input->buf.valid(),
                        "GFX OpenCL: chunked Split input is not materialized for ",
                        m_name);

        for (size_t chunk_slot = 0; chunk_slot < m_artifact.planned_chunks.size();
             ++chunk_slot) {
            const auto& planned_chunk = m_artifact.planned_chunks[chunk_slot];
            const auto& chunk_artifact = *planned_chunk.artifact;
            const uint32_t output_begin = planned_chunk.binding_begin;
            const uint32_t chunk_outputs = planned_chunk.binding_count;
            auto& kernel =
                prepare_planned_chunk_kernel(m_split_chunk_kernels, chunk_slot, chunk_artifact);

            std::vector<KernelArg> args;
            args.reserve(2 + chunk_outputs);
            args.push_back(make_buffer_arg(0, input->buf));
            for (uint32_t local_output = 0; local_output < chunk_outputs; ++local_output) {
                const size_t output_idx = static_cast<size_t>(output_begin + local_output);
                OPENVINO_ASSERT(output_idx < outputs.size() &&
                                    outputs[output_idx] &&
                                    outputs[output_idx]->buf.valid(),
                                "GFX OpenCL: chunked Split output ",
                                output_idx,
                                " is not materialized for ",
                                m_name);
                args.push_back(make_buffer_arg(local_output + 1,
                                               outputs[output_idx]->buf));
            }
            uint32_t count_scalar = count;
            args.push_back(make_bytes_arg(chunk_outputs + 1,
                                          &count_scalar,
                                          sizeof(count_scalar)));

            const size_t local = std::max<size_t>(
                1,
                kernel->clamp_threadgroup_size(chunk_artifact.local_size_hint));
            KernelDispatch dispatch{};
            dispatch.grid[0] = round_up(count, local);
            dispatch.grid[1] = 1;
            dispatch.grid[2] = 1;
            dispatch.threads_per_group[0] = local;
            dispatch.threads_per_group[1] = 1;
            dispatch.threads_per_group[2] = 1;
            kernel->execute(command_buffer, dispatch, args);
        }
        return true;
    }

    std::shared_ptr<ICompiledKernel>& prepare_planned_chunk_kernel(
        std::vector<std::shared_ptr<ICompiledKernel>>& kernels,
        size_t chunk_slot,
        const GfxOpenClSourceArtifact& chunk_artifact) {
        OPENVINO_ASSERT(chunk_artifact.valid,
                        "GFX OpenCL: invalid planned chunk artifact for ",
                        m_name);
        if (kernels.size() <= chunk_slot) {
            kernels.resize(chunk_slot + 1);
        }
        auto& kernel = kernels[chunk_slot];
        if (!kernel) {
            kernel = m_program_cache->get_or_create(
                chunk_artifact.artifact_ref.source_id,
                chunk_artifact.source,
                chunk_artifact.artifact_ref.entry_point,
                gfx_opencl_source_artifact_build_options(chunk_artifact));
            kernel->set_args_count(chunk_artifact.arg_count);
        }
        return kernel;
    }

    void prepare_planned_chunk_kernels(
        std::vector<std::shared_ptr<ICompiledKernel>>& kernels) {
        for (size_t chunk_slot = 0; chunk_slot < m_artifact.planned_chunks.size();
             ++chunk_slot) {
            const auto& planned_chunk = m_artifact.planned_chunks[chunk_slot];
            OPENVINO_ASSERT(planned_chunk.artifact,
                            "GFX OpenCL: missing planned chunk artifact for ",
                            m_name);
            prepare_planned_chunk_kernel(kernels, chunk_slot, *planned_chunk.artifact);
        }
    }

    std::vector<GpuTensor*> resolve_outputs() const {
        if (!m_outputs.empty()) {
            return m_outputs;
        }
        if (m_output) {
            return {m_output};
        }
        return {};
    }

    ov::Shape resolve_output_shape(const GpuTensor& output, size_t output_idx) const {
        if (!output.shape.empty()) {
            return output.shape;
        }
        if (m_node &&
            output_idx < m_node->get_output_size() &&
            m_node->get_output_partial_shape(output_idx).is_static()) {
            return m_node->get_output_shape(output_idx);
        }
        return {};
    }

    ov::Shape resolve_input_shape(size_t input_idx) const {
        if (input_idx < m_inputs.size() && m_inputs[input_idx]) {
            const auto& input = *m_inputs[input_idx];
            if (!input.shape.empty()) {
                return input.shape;
            }
        }
        if (m_node &&
            input_idx < m_node->get_input_size() &&
            m_node->get_input_partial_shape(input_idx).is_static()) {
            return m_node->get_input_shape(input_idx);
        }
        return {};
    }

    GpuTensor* resolve_tensor_input(size_t node_input_idx) {
        if (node_input_idx < m_inputs.size() &&
            m_inputs[node_input_idx] &&
            m_inputs[node_input_idx]->buf.valid()) {
            return m_inputs[node_input_idx];
        }
        return materialize_constant_input(node_input_idx);
    }

    GpuTensor* materialize_constant_input(size_t node_input_idx) {
        if (!m_node || node_input_idx >= m_node->get_input_size()) {
            return nullptr;
        }
        auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(
            m_node->input_value(node_input_idx).get_node_shared_ptr());
        if (!constant) {
            return nullptr;
        }
        if (m_const_inputs.size() < m_node->get_input_size()) {
            m_const_inputs.resize(m_node->get_input_size());
        }
        auto& cached = m_const_inputs[node_input_idx];
        if (cached && cached->buf.valid()) {
            return cached.get();
        }

        OPENVINO_ASSERT(m_buffer_manager,
                        "GFX OpenCL: const input buffer manager is required for ",
                        m_name);
        const auto et = constant->get_output_element_type(0);
        const auto shape = constant->get_output_shape(0);
        const size_t bytes = constant->get_byte_size();
        std::string key = "gfx/opencl_source_const/";
        key += m_name;
        key += "/";
        key += std::to_string(node_input_idx);
        key += "/";
        key += constant->get_friendly_name();
        key += "/";
        key += std::to_string(bytes);

        GpuBuffer buf = m_buffer_manager->wrap_const(key,
                                                     constant->get_data_ptr(),
                                                     bytes,
                                                     et);
        OPENVINO_ASSERT(buf.valid(),
                        "GFX OpenCL: failed to materialize const input slot ",
                        node_input_idx,
                        " for ",
                        m_name);
        cached = std::make_unique<GpuTensor>();
        cached->buf = buf;
        cached->shape = shape;
        cached->expected_type = et;
        cached->prefer_private = false;
        return cached.get();
    }

    ov::Shape resolve_element_count_shape(const std::vector<GpuTensor*>& outputs) const {
        switch (m_artifact.element_count_source) {
            case GfxOpenClSourceElementCountSource::Output0:
                OPENVINO_ASSERT(!outputs.empty() && outputs.front(),
                                "GFX OpenCL: output0 element-count source is missing for ",
                                m_name);
                return resolve_output_shape(*outputs.front(), 0);
            case GfxOpenClSourceElementCountSource::Input0:
                return resolve_input_shape(0);
            default:
                break;
        }
        OPENVINO_THROW("GFX OpenCL: unsupported element-count source for ", m_name);
    }

    bool try_alias_linear_shape_view(const std::vector<GpuTensor*>& outputs) {
        if (!m_node || !is_linear_shape_view_op(m_type)) {
            return false;
        }
        if (outputs.size() != 1 || !outputs.front()) {
            return false;
        }
        GpuTensor* input = resolve_tensor_input(0);
        OPENVINO_ASSERT(input && input->buf.valid(),
                        "GFX OpenCL: missing input buffer for linear view ",
                        m_name);
        OPENVINO_ASSERT(m_node->get_input_partial_shape(0).is_static() &&
                            m_node->get_output_partial_shape(0).is_static(),
                        "GFX OpenCL: linear view requires static input/output shapes for ",
                        m_name);
        const auto input_shape = m_node->get_input_shape(0);
        const auto output_shape = m_node->get_output_shape(0);
        OPENVINO_ASSERT(ov::shape_size(input_shape) == ov::shape_size(output_shape),
                        "GFX OpenCL: linear view element count mismatch for ",
                        m_name);

        GpuTensor* output = outputs.front();
        output->buf = input->buf;
        output->buf.external = true;
        output->buf.owned = false;
        output->shape = output_shape;
        output->expected_type = m_node->get_output_element_type(0);
        output->gqa_broadcast_view = input->gqa_broadcast_view;
        output->gqa_storage_shape = input->gqa_storage_shape;
        output->gqa_kv_heads = input->gqa_kv_heads;
        if (!input->i64_values.empty() &&
            input->i64_values.size() == ov::shape_size(output_shape)) {
            output->i64_values = input->i64_values;
        } else {
            output->i64_values.clear();
        }
        return true;
    }

    bool try_alias_linear_slice_view(const std::vector<GpuTensor*>& outputs) {
        if (!m_node ||
            !(ov::as_type_ptr<const ov::op::v8::Slice>(m_node) ||
              ov::as_type_ptr<const ov::op::v1::StridedSlice>(m_node))) {
            return false;
        }

        RuntimeInputResolver runtime_inputs{&m_inputs, nullptr, nullptr, m_node};
        const auto slice_plan =
            plan_slice_runtime_values(runtime_inputs, outputs, false, m_name);
        if (!slice_plan.linear_view) {
            return false;
        }

        GpuTensor* input = runtime_inputs.tensor(0);
        OPENVINO_ASSERT(input && input->buf.valid(),
                        "GFX OpenCL: missing input buffer for runtime Slice view ",
                        m_name);
        for (auto* out : outputs) {
            if (!out) {
                continue;
            }
            out->buf = input->buf;
            out->buf.external = true;
            out->buf.owned = false;
            out->shape = slice_plan.values.output_shape;
            out->expected_type = slice_plan.values.output_type;
            out->gqa_broadcast_view = input->gqa_broadcast_view;
            out->gqa_storage_shape = input->gqa_storage_shape;
            out->gqa_kv_heads = input->gqa_kv_heads;
            if (!input->i64_values.empty() &&
                input->i64_values.size() == ov::shape_size(out->shape)) {
                out->i64_values = input->i64_values;
            }
        }
        return true;
    }

    std::shared_ptr<const ov::Node> m_node;
    std::shared_ptr<OpenClRuntimeContext> m_context;
    std::shared_ptr<OpenClProgramCache> m_program_cache;
    RuntimeStageExecutableDescriptor m_descriptor;
    GfxOpenClSourceArtifact m_artifact;
    std::shared_ptr<ICompiledKernel> m_kernel;
    std::vector<std::shared_ptr<ICompiledKernel>> m_concat_chunk_kernels;
    std::vector<std::shared_ptr<ICompiledKernel>> m_split_chunk_kernels;
    GpuBufferManager* m_buffer_manager = nullptr;
    std::vector<uint32_t> m_scalar_storage;
    std::vector<GpuTensor*> m_inputs;
    std::vector<GpuTensor*> m_outputs;
    std::vector<std::unique_ptr<GpuTensor>> m_const_inputs;
    GpuTensor* m_output = nullptr;
    std::string m_name;
    std::string m_type;
};

}  // namespace

std::unique_ptr<GpuStage> create_opencl_source_stage(
    const std::shared_ptr<const ov::Node>& node,
    std::shared_ptr<OpenClRuntimeContext> context,
    RuntimeStageExecutableDescriptor descriptor,
    GfxOpenClSourceArtifact artifact) {
    return std::make_unique<OpenClSourceStage>(node,
                                               std::move(context),
                                               std::move(descriptor),
                                               std::move(artifact));
}

}  // namespace gfx_plugin
}  // namespace ov
