// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/runtime/opencl_source_stage.hpp"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backends/opencl/runtime/opencl_program_cache.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "runtime/gpu_backend_base.hpp"

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
                                            GfxOpenClBaselineOp op,
                                            GfxOpenClBaselineInputMode input_mode,
                                            float scalar_constant_f32,
                                            const std::vector<uint32_t>& static_u32_scalars,
                                            size_t& static_u32_idx) {
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

class OpenClSourceStage final : public GpuStage {
public:
    OpenClSourceStage(std::shared_ptr<const ov::Node> node,
                      std::shared_ptr<OpenClRuntimeContext> context,
                      GfxOpenClSourceArtifact artifact)
        : m_node(std::move(node)),
          m_context(std::move(context)),
          m_artifact(std::move(artifact)) {
        OPENVINO_ASSERT(m_node, "GFX OpenCL: source stage requires a node");
        OPENVINO_ASSERT(m_context, "GFX OpenCL: source stage requires a runtime context");
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
        m_kernel = m_program_cache->get_or_create(m_artifact.artifact_ref.source_id,
                                                  m_artifact.source,
                                                  m_artifact.artifact_ref.entry_point,
                                                  gfx_opencl_source_artifact_build_options(m_artifact));
        m_kernel->set_args_count(m_artifact.arg_count);
    }

    void execute(GpuCommandBufferHandle command_buffer) override {
        if (!m_kernel) {
            compile(m_buffer_manager);
        }
        const auto outputs = resolve_outputs();
        OPENVINO_ASSERT(outputs.size() == m_artifact.direct_output_count,
                        "GFX OpenCL: output binding count does not match source artifact ABI for ",
                        m_name);
        OPENVINO_ASSERT(!outputs.empty(),
                        "GFX OpenCL: source artifact must bind at least one output for ",
                        m_name);
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
                                output_type == ov::element::f32 ||
                                output_type == ov::element::boolean ||
                                output_type == ov::element::i32 ||
                                output_type == ov::element::i64,
                            "GFX OpenCL: baseline stage currently supports f32, boolean, i32 and i64 outputs only");
        }
        const auto count = checked_element_count(resolve_element_count_shape(outputs),
                                                 "GFX OpenCL");
        OPENVINO_ASSERT(count > 0, "GFX OpenCL: zero-sized baseline dispatch is not supported yet");

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
        size_t input_idx = 0;
        size_t output_idx = 0;
        size_t scalar_idx = 0;
        size_t static_u32_idx = 0;
        for (size_t arg_idx = 0; arg_idx < roles.size(); ++arg_idx) {
            switch (roles[arg_idx]) {
                case GfxKernelBufferRole::TensorInput: {
                    OPENVINO_ASSERT(input_idx < m_artifact.direct_input_indices.size(),
                                    "GFX OpenCL: tensor ABI has no input slot mapping for ",
                                    m_name);
                    const size_t node_input_idx = m_artifact.direct_input_indices[input_idx];
                    OPENVINO_ASSERT(node_input_idx < m_inputs.size() &&
                                        m_inputs[node_input_idx] &&
                                        m_inputs[node_input_idx]->buf.valid(),
                                    "GFX OpenCL: input slot ",
                                    node_input_idx,
                                    " is not materialized for ",
                                    m_name);
                    args.push_back(make_buffer_arg(static_cast<uint32_t>(arg_idx),
                                                   m_inputs[node_input_idx]->buf));
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
                        m_artifact.static_u32_scalars,
                        static_u32_idx));
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

        const size_t local = std::max<size_t>(1, m_kernel->clamp_threadgroup_size(m_artifact.baseline_local_size));
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
        auto cloned = std::make_unique<OpenClSourceStage>(m_node, m_context, m_artifact);
        cloned->m_name = m_name;
        cloned->m_type = m_type;
        cloned->m_program_cache = m_program_cache;
        if (m_kernel) {
            cloned->m_kernel = m_kernel->fork();
        }
        return cloned;
    }

private:
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

    std::shared_ptr<const ov::Node> m_node;
    std::shared_ptr<OpenClRuntimeContext> m_context;
    std::shared_ptr<OpenClProgramCache> m_program_cache;
    GfxOpenClSourceArtifact m_artifact;
    std::shared_ptr<ICompiledKernel> m_kernel;
    GpuBufferManager* m_buffer_manager = nullptr;
    std::vector<uint32_t> m_scalar_storage;
    std::vector<GpuTensor*> m_inputs;
    std::vector<GpuTensor*> m_outputs;
    GpuTensor* m_output = nullptr;
    std::string m_name;
    std::string m_type;
};

}  // namespace

std::unique_ptr<GpuStage> create_opencl_source_stage(
    const std::shared_ptr<const ov::Node>& node,
    std::shared_ptr<OpenClRuntimeContext> context) {
    auto artifact = resolve_gfx_opencl_source_artifact(node);
    if (!artifact || !artifact->valid) {
        return {};
    }
    return std::make_unique<OpenClSourceStage>(node, std::move(context), std::move(*artifact));
}

}  // namespace gfx_plugin
}  // namespace ov
