// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/runtime/opencl_source_stage.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
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
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "runtime/descriptor_const_tensor_materializer.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_stage_runtime_values.hpp"
#include "runtime/gpu_backend_base.hpp"
#include "runtime/kernel_launch_plan.hpp"
#include "runtime/tensor_binding_contract.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

uint32_t checked_element_count(const ov::Shape &shape, const char *label) {
  const auto elements = ov::shape_size(shape);
  OPENVINO_ASSERT(
      elements <= std::numeric_limits<uint32_t>::max(), label,
      ": OpenCL baseline kernel supports at most uint32 element counts");
  return static_cast<uint32_t>(elements);
}

uint32_t scalar_value_for_opencl_source_arg(
    GfxOpenClSourceScalarArg scalar, uint32_t element_count,
    GfxOpenClArtifactOp op, GfxOpenClArtifactInputMode input_mode,
    float scalar_constant_f32, const std::vector<ov::Shape> &input_shapes,
    const ov::Shape &output0_shape,
    const std::vector<uint32_t> &static_u32_scalars,
    const std::vector<float> &static_f32_scalars, size_t &static_u32_idx,
    size_t &static_f32_idx) {
  const auto raw_scalar = static_cast<uint32_t>(scalar);
  auto resolve_dim =
      [&](GfxOpenClSourceScalarArg base, size_t shape_idx,
          const std::vector<ov::Shape> &shapes) -> std::optional<uint32_t> {
    const auto base_value = static_cast<uint32_t>(base);
    if (raw_scalar < base_value || raw_scalar >= base_value + 8u) {
      return std::nullopt;
    }
    const size_t axis = static_cast<size_t>(raw_scalar - base_value);
    if (shape_idx >= shapes.size() || axis >= shapes[shape_idx].size()) {
      return 0;
    }
    OPENVINO_ASSERT(shapes[shape_idx][axis] <=
                        std::numeric_limits<uint32_t>::max(),
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
  if (raw_scalar >=
          static_cast<uint32_t>(GfxOpenClSourceScalarArg::Output0Dim0) &&
      raw_scalar <=
          static_cast<uint32_t>(GfxOpenClSourceScalarArg::Output0Dim7)) {
    const size_t axis = static_cast<size_t>(
        raw_scalar -
        static_cast<uint32_t>(GfxOpenClSourceScalarArg::Output0Dim0));
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

ov::element::Type
descriptor_output_type(const RuntimeStageExecutableDescriptor &descriptor,
                       size_t output_idx) {
  if (output_idx >= descriptor.output_bindings.size()) {
    return ov::element::dynamic;
  }
  return element_type_from_contract(
      descriptor.output_bindings[output_idx].element_type);
}

bool descriptor_output_shape(const RuntimeStageExecutableDescriptor &descriptor,
                             size_t output_idx, ov::Shape &shape) {
  if (output_idx >= descriptor.output_bindings.size()) {
    return false;
  }
  return parse_static_shape_contract(
      descriptor.output_bindings[output_idx].partial_shape, shape);
}

bool descriptor_input_shape(const RuntimeStageExecutableDescriptor &descriptor,
                            size_t input_idx, ov::Shape &shape) {
  if (input_idx >= descriptor.input_bindings.size()) {
    return false;
  }
  return parse_static_shape_contract(
      descriptor.input_bindings[input_idx].partial_shape, shape);
}

OpenClProgramBuildRequest make_opencl_program_build_request(
    const RuntimeStageExecutableDescriptor &descriptor,
    const GfxOpenClSourceArtifact &artifact) {
  OpenClProgramBuildRequest request;
  request.manifest_ref = descriptor.manifest_ref;
  request.abi_fingerprint = descriptor.abi_fingerprint;
  request.artifact_key = descriptor.artifact_key;
  request.backend_domain = descriptor.backend_domain;
  request.kernel_id = descriptor.kernel_id;
  request.stage_record_key = descriptor.stage_record_key;
  request.source_id = artifact.artifact_ref.source_id;
  request.source = artifact.source;
  request.entry_point = artifact.artifact_ref.entry_point;
  request.compile_options_key = descriptor.compile_options_key;
  request.build_options = gfx_opencl_source_artifact_build_options(artifact);
  return request;
}

class OpenClSourceStage final : public GpuStage {
public:
  OpenClSourceStage(std::shared_ptr<OpenClRuntimeContext> context,
                    RuntimeStageExecutableDescriptor descriptor,
                    GfxOpenClSourceArtifact artifact)
      : m_context(std::move(context)), m_descriptor(std::move(descriptor)),
        m_artifact(std::move(artifact)) {
    OPENVINO_ASSERT(m_context,
                    "GFX OpenCL: source stage requires a runtime context");
    OPENVINO_ASSERT(
        m_descriptor.payload_kind == KernelArtifactPayloadKind::OpenClSource,
        "GFX OpenCL: source stage requires OpenCL source runtime descriptor");
    OPENVINO_ASSERT(m_descriptor.backend_domain == "opencl",
                    "GFX OpenCL: source stage descriptor backend domain drift");
    OPENVINO_ASSERT(m_descriptor.entry_point ==
                        m_artifact.artifact_ref.entry_point,
                    "GFX OpenCL: source stage descriptor entry point drift");
    OPENVINO_ASSERT(m_descriptor.launch_plan.valid,
                    "GFX OpenCL: source stage descriptor launch plan is "
                    "required");
    OPENVINO_ASSERT(!m_descriptor.stage_name.empty(),
                    "GFX OpenCL: source stage descriptor must provide stage "
                    "name");
    OPENVINO_ASSERT(!m_descriptor.op_family.empty(),
                    "GFX OpenCL: source stage descriptor must provide op "
                    "family");
    m_program_cache = std::make_shared<OpenClProgramCache>(m_context);
    m_name = m_descriptor.stage_name;
    m_type = m_descriptor.op_family;
  }

  void init(GpuBufferManager *buffer_manager) override {
    m_buffer_manager = buffer_manager;
  }

  void prepare_runtime_handle(GpuBufferManager *buffer_manager) override {
    if (buffer_manager) {
      m_buffer_manager = buffer_manager;
    }
    OPENVINO_ASSERT(m_artifact.valid,
                    "GFX OpenCL: invalid source artifact for ", m_type);
    if (m_kernel) {
      return;
    }
    if (has_planned_source_dispatches()) {
      prepare_planned_source_dispatch_kernels();
      return;
    }
    m_kernel = m_program_cache->get_or_create(
        make_opencl_program_build_request(m_descriptor, m_artifact));
    m_kernel->set_args_count(m_descriptor.abi_arg_count);
  }

  void execute(GpuCommandBufferHandle command_buffer) override {
    if (has_planned_source_dispatches()) {
      OPENVINO_ASSERT(
          planned_source_dispatch_kernels_ready(),
          "GFX OpenCL: planned source dispatch handles are not prepared for ",
          m_name);
    } else {
      OPENVINO_ASSERT(
          m_kernel, "GFX OpenCL: runtime handle is not prepared for ", m_name);
    }
    const auto outputs = resolve_outputs();
    OPENVINO_ASSERT(outputs.size() == m_descriptor.abi_output_arg_count,
                    "GFX OpenCL: output binding count does not match source "
                    "descriptor ABI for ",
                    m_name);
    OPENVINO_ASSERT(
        !outputs.empty(),
        "GFX OpenCL: source artifact must bind at least one output for ",
        m_name);
    if (try_alias_linear_shape_view(outputs) ||
        try_alias_linear_slice_view(outputs)) {
      return;
    }
    for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
      GpuTensor *output = outputs[output_idx];
      OPENVINO_ASSERT(output && output->buf.valid(),
                      "GFX OpenCL: output buffer ", output_idx,
                      " is not materialized for ", m_name);
      auto output_type = output->expected_type;
      if (output_type == ov::element::dynamic) {
        output_type = descriptor_output_type(m_descriptor, output_idx);
      }
      OPENVINO_ASSERT(output_type == ov::element::dynamic ||
                          output_type == ov::element::f16 ||
                          output_type == ov::element::f32 ||
                          output_type == ov::element::boolean ||
                          output_type == ov::element::i32 ||
                          output_type == ov::element::i64,
                      "GFX OpenCL: source stage currently supports f16, "
                      "f32, boolean, i32 and i64 outputs only");
    }
    const auto count = checked_element_count(
        resolve_element_count_shape(outputs), "GFX OpenCL");
    OPENVINO_ASSERT(
        count > 0,
        "GFX OpenCL: zero-sized baseline dispatch is not supported yet");

    if (has_planned_source_dispatches()) {
      execute_planned_source_dispatches(command_buffer, outputs, count);
      return;
    }

    execute_source_artifact_dispatch(command_buffer, m_artifact, *m_kernel,
                                     outputs, count);
  }

  void set_inputs(const std::vector<GpuTensor *> &inputs) override {
    m_inputs = inputs;
  }

  void set_output(GpuTensor *output) override {
    m_output = output;
    m_outputs.clear();
    if (output) {
      m_outputs.push_back(output);
    }
  }

  void set_output_refs(const std::vector<GpuTensor *> &outputs) override {
    m_outputs = outputs;
    m_output = outputs.empty() ? nullptr : outputs.front();
  }

  const std::string &name() const override { return m_name; }
  const std::string &type() const override { return m_type; }

  std::unique_ptr<GpuStage> clone() const override {
    auto cloned = std::make_unique<OpenClSourceStage>(m_context, m_descriptor,
                                                      m_artifact);
    cloned->m_name = m_name;
    cloned->m_type = m_type;
    cloned->m_program_cache = m_program_cache;
    if (m_kernel) {
      cloned->m_kernel = m_kernel->fork();
    }
    cloned->m_planned_source_dispatch_kernels.reserve(
        m_planned_source_dispatch_kernels.size());
    for (const auto &kernel : m_planned_source_dispatch_kernels) {
      cloned->m_planned_source_dispatch_kernels.push_back(kernel ? kernel->fork()
                                                                : nullptr);
    }
    return cloned;
  }

private:
  bool has_planned_source_dispatches() const {
    return !m_artifact.planned_chunks.empty();
  }

  std::shared_ptr<ICompiledKernel> &prepare_planned_source_dispatch_kernel(
      size_t dispatch_slot, const GfxOpenClSourceArtifact &chunk_artifact) {
    OPENVINO_ASSERT(chunk_artifact.valid,
                    "GFX OpenCL: invalid planned source artifact for ", m_name);
    if (m_planned_source_dispatch_kernels.size() <= dispatch_slot) {
      m_planned_source_dispatch_kernels.resize(dispatch_slot + 1);
    }
    auto &kernel = m_planned_source_dispatch_kernels[dispatch_slot];
    if (!kernel) {
      kernel = m_program_cache->get_or_create(
          make_opencl_program_build_request(m_descriptor, chunk_artifact));
      kernel->set_args_count(chunk_artifact.arg_count);
    }
    return kernel;
  }

  std::shared_ptr<ICompiledKernel> &prepared_planned_source_dispatch_kernel(
      size_t dispatch_slot, const GfxOpenClSourceArtifact &chunk_artifact) {
    OPENVINO_ASSERT(
        dispatch_slot < m_planned_source_dispatch_kernels.size() &&
            m_planned_source_dispatch_kernels[dispatch_slot],
        "GFX OpenCL: runtime handle is not prepared for planned source dispatch ",
        chunk_artifact.artifact_ref.entry_point, " in ", m_name);
    return m_planned_source_dispatch_kernels[dispatch_slot];
  }

  bool planned_source_dispatch_kernels_ready() const {
    if (m_planned_source_dispatch_kernels.size() !=
        m_artifact.planned_chunks.size()) {
      return false;
    }
    for (const auto &kernel : m_planned_source_dispatch_kernels) {
      if (!kernel) {
        return false;
      }
    }
    return true;
  }

  void prepare_planned_source_dispatch_kernels() {
    for (size_t dispatch_slot = 0;
         dispatch_slot < m_artifact.planned_chunks.size(); ++dispatch_slot) {
      const auto &planned = m_artifact.planned_chunks[dispatch_slot];
      OPENVINO_ASSERT(planned.artifact,
                      "GFX OpenCL: missing planned source artifact for ",
                      m_name);
      prepare_planned_source_dispatch_kernel(dispatch_slot, *planned.artifact);
    }
  }

  uint32_t planned_source_dispatch_element_count(
      const GfxOpenClSourceChunkArtifact &planned,
      uint32_t base_element_count) const {
    OPENVINO_ASSERT(planned.element_count_multiplier != 0 &&
                        planned.element_count_divisor != 0,
                    "GFX OpenCL: planned source dispatch has invalid "
                    "element-count scale for ",
                    m_name);
    OPENVINO_ASSERT(
        base_element_count % planned.element_count_divisor == 0,
        "GFX OpenCL: planned source dispatch element-count divisor drift for ",
        m_name);
    const uint64_t scaled =
        (static_cast<uint64_t>(base_element_count) /
         planned.element_count_divisor) *
        planned.element_count_multiplier;
    OPENVINO_ASSERT(scaled > 0 &&
                        scaled <= std::numeric_limits<uint32_t>::max(),
                    "GFX OpenCL: planned source dispatch element count exceeds "
                    "OpenCL scalar range for ",
                    m_name);
    return static_cast<uint32_t>(scaled);
  }

  std::vector<GpuTensor *> planned_source_dispatch_outputs(
      const GfxOpenClSourceChunkArtifact &planned,
      const std::vector<GpuTensor *> &outputs) const {
    if (planned.binding_role ==
        GfxOpenClSourceChunkBindingRole::DirectInputs) {
      return outputs;
    }
    OPENVINO_ASSERT(planned.binding_role ==
                        GfxOpenClSourceChunkBindingRole::DirectOutputs,
                    "GFX OpenCL: planned source dispatch has unknown binding "
                    "role for ",
                    m_name);
    const size_t output_begin = planned.binding_begin;
    const size_t output_end =
        output_begin + static_cast<size_t>(planned.binding_count);
    OPENVINO_ASSERT(output_begin < output_end && output_end <= outputs.size(),
                    "GFX OpenCL: planned source dispatch output binding range "
                    "drift for ",
                    m_name);
    return std::vector<GpuTensor *>(outputs.begin() + output_begin,
                                    outputs.begin() + output_end);
  }

  std::vector<GfxKernelBufferRole>
  source_artifact_launch_roles(const GfxOpenClSourceArtifact &artifact) const {
    OPENVINO_ASSERT(artifact.stage_manifest.valid &&
                        artifact.stage_manifest.custom_kernel.valid &&
                        artifact.stage_manifest.custom_kernel.external_buffer_abi
                            .valid,
                    "GFX OpenCL: planned source artifact launch ABI is missing "
                    "for ",
                    m_name);
    auto roles = materialize_gfx_kernel_external_buffer_roles(
        artifact.stage_manifest.custom_kernel.external_buffer_abi);
    OPENVINO_ASSERT(!roles.empty(),
                    "GFX OpenCL: planned source artifact launch roles are empty "
                    "for ",
                    m_name);
    return roles;
  }

  void execute_source_artifact_dispatch(
      GpuCommandBufferHandle command_buffer,
      const GfxOpenClSourceArtifact &artifact, ICompiledKernel &kernel,
      const std::vector<GpuTensor *> &outputs, uint32_t count) {
    const auto roles = source_artifact_launch_roles(artifact);
    OPENVINO_ASSERT(
        artifact.arg_count == roles.size(),
        "GFX OpenCL: planned source artifact arg count does not match role ABI "
        "for ",
        m_name);
    const auto tensor_input_count = static_cast<size_t>(std::count(
        roles.begin(), roles.end(), GfxKernelBufferRole::TensorInput));
    OPENVINO_ASSERT(
        artifact.direct_input_indices.size() == tensor_input_count,
        "GFX OpenCL: planned source artifact direct input mapping is "
        "inconsistent for ",
        m_name);

    std::vector<ov::Shape> input_shapes;
    input_shapes.reserve(std::min<size_t>(artifact.direct_input_indices.size(), 4));
    for (size_t idx = 0; idx < artifact.direct_input_indices.size() && idx < 4;
         ++idx) {
      input_shapes.push_back(resolve_input_shape(artifact.direct_input_indices[idx]));
    }
    const ov::Shape output0_shape = resolve_output_shape(*outputs.front(), 0);
    size_t static_u32_idx = 0;
    size_t static_f32_idx = 0;
    std::vector<uint32_t> scalar_values;
    scalar_values.reserve(artifact.scalar_args.size());
    for (const auto scalar_kind : artifact.scalar_args) {
      scalar_values.push_back(scalar_value_for_opencl_source_arg(
          scalar_kind, count, artifact.op, artifact.input_mode,
          artifact.scalar_constant_f32, input_shapes, output0_shape,
          artifact.static_u32_scalars, artifact.static_f32_scalars,
          static_u32_idx, static_f32_idx));
    }
    OPENVINO_ASSERT(static_u32_idx == artifact.static_u32_scalars.size(),
                    "GFX OpenCL: not all planned source static u32 scalars were "
                    "consumed for ",
                    m_name);
    OPENVINO_ASSERT(static_f32_idx == artifact.static_f32_scalars.size(),
                    "GFX OpenCL: not all planned source static f32 scalars were "
                    "consumed for ",
                    m_name);

    const auto const_tensor_args =
        materialize_const_tensor_args(static_cast<size_t>(std::count(
            roles.begin(), roles.end(), GfxKernelBufferRole::ConstTensor)));
    const size_t runtime_param_count = static_cast<size_t>(std::count(
        roles.begin(), roles.end(), GfxKernelBufferRole::RuntimeParams));
    m_kernel_extra_inputs = materialize_runtime_param_args(
        outputs, runtime_param_count, &artifact.direct_input_indices);
    auto launch_plan = build_role_ordered_kernel_launch_plan<uint32_t>(
        roles, artifact.direct_input_indices, scalar_values, outputs,
        const_tensor_args, m_kernel_extra_inputs,
        [&](size_t node_input_idx) {
          return resolve_tensor_input(node_input_idx);
        },
        m_name);
    m_scalar_storage = launch_plan.scalar_storage;

    if (gfx_log_debug_enabled()) {
      std::ostringstream oss;
      oss << "source_stage name=" << m_name
          << " entry=" << artifact.artifact_ref.entry_point
          << " count=" << count << " inputs=" << input_shapes.size()
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
        const auto &shape = input_shapes[input_idx];
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

    const size_t local =
        std::max<size_t>(1, kernel.clamp_threadgroup_size(artifact.local_size_hint));
    KernelDispatch dispatch{};
    dispatch.grid[0] = round_up(count, local);
    dispatch.grid[1] = 1;
    dispatch.grid[2] = 1;
    dispatch.threads_per_group[0] = local;
    dispatch.threads_per_group[1] = 1;
    dispatch.threads_per_group[2] = 1;
    kernel.execute(command_buffer, dispatch, launch_plan.args);
  }

  void execute_planned_source_dispatches(
      GpuCommandBufferHandle command_buffer,
      const std::vector<GpuTensor *> &outputs, uint32_t base_count) {
    for (size_t dispatch_slot = 0;
         dispatch_slot < m_artifact.planned_chunks.size(); ++dispatch_slot) {
      const auto &planned = m_artifact.planned_chunks[dispatch_slot];
      OPENVINO_ASSERT(planned.artifact,
                      "GFX OpenCL: planned source dispatch artifact is missing "
                      "for ",
                      m_name);
      const auto &artifact = *planned.artifact;
      auto &kernel =
          prepared_planned_source_dispatch_kernel(dispatch_slot, artifact);
      auto planned_outputs = planned_source_dispatch_outputs(planned, outputs);
      const uint32_t count =
          planned_source_dispatch_element_count(planned, base_count);
      execute_source_artifact_dispatch(command_buffer, artifact, *kernel,
                                       planned_outputs, count);
    }
  }

  std::vector<GpuTensor *> resolve_outputs() const {
    if (!m_outputs.empty()) {
      return m_outputs;
    }
    if (m_output) {
      return {m_output};
    }
    return {};
  }

  ov::Shape resolve_output_shape(const GpuTensor &output,
                                 size_t output_idx) const {
    if (!output.shape.empty()) {
      return output.shape;
    }
    ov::Shape descriptor_shape;
    if (descriptor_output_shape(m_descriptor, output_idx, descriptor_shape)) {
      return descriptor_shape;
    }
    return {};
  }

  ov::Shape resolve_input_shape(size_t input_idx) const {
    if (input_idx < m_inputs.size() && m_inputs[input_idx]) {
      const auto &input = *m_inputs[input_idx];
      if (!input.shape.empty()) {
        return input.shape;
      }
    }
    ov::Shape descriptor_shape;
    if (descriptor_input_shape(m_descriptor, input_idx, descriptor_shape)) {
      return descriptor_shape;
    }
    return {};
  }

  std::vector<GpuTensor>
  materialize_runtime_param_args(
      const std::vector<GpuTensor *> &outputs, size_t runtime_param_count,
      const std::vector<size_t> *direct_input_indices) {
    if (runtime_param_count == 0) {
      return {};
    }
    OPENVINO_ASSERT(m_buffer_manager,
                    "GFX OpenCL: runtime-param buffer manager is required for ",
                    m_name);
    OPENVINO_ASSERT(
        !outputs.empty() && outputs.front(),
        "GFX OpenCL: runtime-param materialization requires output0 "
        "for ",
        m_name);

    RuntimeInputResolver runtime_inputs;
    runtime_inputs.inputs = &m_inputs;
    runtime_inputs.descriptor = &m_descriptor;
    const std::vector<int32_t> no_compiler_scalar_args;
    auto materialization = materialize_descriptor_owned_runtime_param_payload(
        *m_buffer_manager, m_descriptor, runtime_inputs, outputs,
        runtime_param_count, no_compiler_scalar_args, m_name,
        direct_input_indices);
    if (!materialization.available) {
      OPENVINO_ASSERT(
          false, "GFX OpenCL: RuntimeParams ABI is not descriptor-owned for ",
          m_name,
          "; compiler descriptor/artifact metadata must own runtime payload "
          "construction");
    }
    return std::move(materialization.extra_inputs);
  }

  GpuTensor *resolve_tensor_input(size_t node_input_idx) {
    if (node_input_idx < m_inputs.size() && m_inputs[node_input_idx] &&
        m_inputs[node_input_idx]->buf.valid()) {
      return m_inputs[node_input_idx];
    }
    prepare_constant_input_buffers();
    if (node_input_idx < m_const_inputs.buffers.size() &&
        node_input_idx < m_const_inputs.present.size() &&
        m_const_inputs.present[node_input_idx] &&
        m_const_inputs.buffers[node_input_idx].buf.valid()) {
      return &m_const_inputs.buffers[node_input_idx];
    }
    return nullptr;
  }

  void prepare_constant_input_buffers() {
    if (m_const_inputs_materialized || m_descriptor.const_tensors.empty()) {
      return;
    }
    OPENVINO_ASSERT(m_buffer_manager,
                    "GFX OpenCL: const input buffer manager is required for ",
                    m_name);
    m_const_inputs = materialize_descriptor_const_tensor_slots(
        *m_buffer_manager, m_descriptor, "opencl/source");
    m_const_inputs_materialized = true;
  }

  std::vector<GpuTensor *>
  materialize_const_tensor_args(size_t expected_count) {
    std::vector<GpuTensor *> tensors;
    if (expected_count == 0) {
      return tensors;
    }
    prepare_constant_input_buffers();
    tensors = descriptor_const_tensor_args(m_const_inputs, expected_count);
    OPENVINO_ASSERT(tensors.size() == expected_count,
                    "GFX OpenCL: ConstTensor ABI count does not match "
                    "descriptor-owned constants for ",
                    m_name);
    return tensors;
  }

  ov::Shape
  resolve_element_count_shape(const std::vector<GpuTensor *> &outputs) const {
    switch (m_artifact.element_count_source) {
    case GfxOpenClSourceElementCountSource::Output0:
      OPENVINO_ASSERT(
          !outputs.empty() && outputs.front(),
          "GFX OpenCL: output0 element-count source is missing for ", m_name);
      return resolve_output_shape(*outputs.front(), 0);
    case GfxOpenClSourceElementCountSource::Input0:
      return resolve_input_shape(0);
    default:
      break;
    }
    OPENVINO_THROW("GFX OpenCL: unsupported element-count source for ", m_name);
  }

  bool try_alias_linear_shape_view(const std::vector<GpuTensor *> &outputs) {
    if (!is_linear_shape_view_op(m_type)) {
      return false;
    }
    if (outputs.size() != 1 || !outputs.front()) {
      return false;
    }
    GpuTensor *input = resolve_tensor_input(0);
    OPENVINO_ASSERT(input && input->buf.valid(),
                    "GFX OpenCL: missing input buffer for linear view ",
                    m_name);
    ov::Shape input_shape;
    ov::Shape output_shape;
    OPENVINO_ASSERT(descriptor_input_shape(m_descriptor, 0, input_shape) &&
                        descriptor_output_shape(m_descriptor, 0, output_shape),
                    "GFX OpenCL: linear view requires descriptor-owned static "
                    "input/output shapes for ",
                    m_name);
    OPENVINO_ASSERT(ov::shape_size(input_shape) == ov::shape_size(output_shape),
                    "GFX OpenCL: linear view element count mismatch for ",
                    m_name);

    GpuTensor *output = outputs.front();
    output->buf = input->buf;
    output->buf.external = true;
    output->buf.owned = false;
    output->shape = output_shape;
    output->expected_type = descriptor_output_type(m_descriptor, 0);
    if (output->expected_type == ov::element::dynamic) {
      output->expected_type = input->expected_type;
    }
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

  bool try_alias_linear_slice_view(const std::vector<GpuTensor *> &outputs) {
    (void)outputs;
    return false;
  }

  std::shared_ptr<OpenClRuntimeContext> m_context;
  std::shared_ptr<OpenClProgramCache> m_program_cache;
  RuntimeStageExecutableDescriptor m_descriptor;
  GfxOpenClSourceArtifact m_artifact;
  std::shared_ptr<ICompiledKernel> m_kernel;
  std::vector<std::shared_ptr<ICompiledKernel>>
      m_planned_source_dispatch_kernels;
  GpuBufferManager *m_buffer_manager = nullptr;
  std::vector<uint32_t> m_scalar_storage;
  std::vector<GpuTensor *> m_inputs;
  std::vector<GpuTensor *> m_outputs;
  DescriptorConstTensorSlots m_const_inputs;
  bool m_const_inputs_materialized = false;
  std::vector<GpuTensor> m_kernel_extra_inputs;
  GpuTensor *m_output = nullptr;
  std::string m_name;
  std::string m_type;
};

} // namespace

std::unique_ptr<GpuStage>
create_opencl_source_stage(std::shared_ptr<OpenClRuntimeContext> context,
                           RuntimeStageExecutableDescriptor descriptor,
                           GfxOpenClSourceArtifact artifact) {
  return std::make_unique<OpenClSourceStage>(
      std::move(context), std::move(descriptor), std::move(artifact));
}

} // namespace gfx_plugin
} // namespace ov
