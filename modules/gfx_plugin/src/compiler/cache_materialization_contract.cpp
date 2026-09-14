// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/cache_materialization_contract.hpp"

#include <sstream>
#include <utility>

#include "compiler/cache_materialization_wire.hpp"
#include "runtime/tensor_binding_contract.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

using WireReader = detail::MaterializationWireReader;
using detail::append_materialization_bool;
using detail::append_materialization_field;
using detail::append_materialization_float_vector;
using detail::append_materialization_integral_vector;
using detail::append_materialization_number;
using detail::append_materialization_string_vector;

void append_field(std::ostringstream &os, std::string_view value) {
  append_materialization_field(os, value);
}

void append_bool(std::ostringstream &os, bool value) {
  append_materialization_bool(os, value);
}

template <typename T>
void append_number(std::ostringstream &os, T value) {
  append_materialization_number(os, value);
}

template <typename T>
void append_integral_vector(std::ostringstream &os,
                            const std::vector<T> &values) {
  append_materialization_integral_vector(os, values);
}

void append_float_vector(std::ostringstream &os,
                         const std::vector<float> &values) {
  append_materialization_float_vector(os, values);
}

void append_string_vector(std::ostringstream &os,
                          const std::vector<std::string> &values) {
  append_materialization_string_vector(os, values);
}

std::string shape_to_contract(const ov::Shape &shape) {
  std::ostringstream os;
  os << "{";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) {
      os << ",";
    }
    os << shape[i];
  }
  os << "}";
  return os.str();
}

ov::Shape read_shape(WireReader &reader, std::string_view name) {
  ov::Shape shape;
  const auto contract = reader.string_field(name);
  if (!contract.empty() && !parse_static_shape_contract(contract, shape)) {
    reader.diagnostic(std::string("cache materialization field ") +
                      std::string(name) + " is not a static shape");
  }
  return shape;
}

ov::element::Type read_element_type(WireReader &reader, std::string_view name) {
  const auto contract = reader.string_field(name);
  const auto type = element_type_from_contract(contract);
  if (!contract.empty() && type == ov::element::dynamic) {
    reader.diagnostic(std::string("cache materialization field ") +
                      std::string(name) + " is not an element type");
  }
  return type;
}

std::string_view origin_to_string(KernelArtifactOrigin origin) noexcept {
  return kernel_artifact_origin_to_string(origin);
}

KernelArtifactOrigin origin_from_string(std::string_view value) noexcept {
  if (value == "common") {
    return KernelArtifactOrigin::Common;
  }
  if (value == "metadata") {
    return KernelArtifactOrigin::Metadata;
  }
  if (value == "vendor_primitive") {
    return KernelArtifactOrigin::VendorPrimitive;
  }
  if (value == "generated") {
    return KernelArtifactOrigin::Generated;
  }
  if (value == "handwritten_exception") {
    return KernelArtifactOrigin::HandwrittenException;
  }
  return KernelArtifactOrigin::Unknown;
}

std::string_view payload_kind_to_string(KernelArtifactPayloadKind kind) noexcept {
  return kernel_artifact_payload_kind_to_string(kind);
}

KernelArtifactPayloadKind payload_kind_from_string(std::string_view value) noexcept {
  if (value == "vendor_descriptor") {
    return KernelArtifactPayloadKind::VendorDescriptor;
  }
  if (value == "msl_source") {
    return KernelArtifactPayloadKind::MslSource;
  }
  if (value == "opencl_source") {
    return KernelArtifactPayloadKind::OpenClSource;
  }
  return KernelArtifactPayloadKind::None;
}

std::string_view tensor_ref_kind_to_string(PipelineStageTensorRefKind kind) noexcept {
  switch (kind) {
  case PipelineStageTensorRefKind::Parameter:
    return "parameter";
  case PipelineStageTensorRefKind::StageOutput:
    return "stage_output";
  case PipelineStageTensorRefKind::None:
  default:
    return "none";
  }
}

PipelineStageTensorRefKind tensor_ref_kind_from_string(std::string_view value) noexcept {
  if (value == "parameter") {
    return PipelineStageTensorRefKind::Parameter;
  }
  if (value == "stage_output") {
    return PipelineStageTensorRefKind::StageOutput;
  }
  return PipelineStageTensorRefKind::None;
}

std::string_view public_output_kind_to_string(
    RuntimePublicOutputSourceKind kind) noexcept {
  switch (kind) {
  case RuntimePublicOutputSourceKind::Parameter:
    return "parameter";
  case RuntimePublicOutputSourceKind::StageOutput:
    return "stage_output";
  case RuntimePublicOutputSourceKind::None:
  default:
    return "none";
  }
}

RuntimePublicOutputSourceKind public_output_kind_from_string(
    std::string_view value) noexcept {
  if (value == "parameter") {
    return RuntimePublicOutputSourceKind::Parameter;
  }
  if (value == "stage_output") {
    return RuntimePublicOutputSourceKind::StageOutput;
  }
  return RuntimePublicOutputSourceKind::None;
}

std::string_view materialization_kind_to_string(
    PipelineStageMaterializationKind kind) noexcept {
  switch (kind) {
  case PipelineStageMaterializationKind::VendorAttention:
    return "vendor_attention";
  case PipelineStageMaterializationKind::FusedAttentionSequence:
    return "fused_attention_sequence";
  case PipelineStageMaterializationKind::SingleStage:
  default:
    return "single_stage";
  }
}

PipelineStageMaterializationKind materialization_kind_from_string(
    std::string_view value) noexcept {
  if (value == "vendor_attention") {
    return PipelineStageMaterializationKind::VendorAttention;
  }
  if (value == "fused_attention_sequence") {
    return PipelineStageMaterializationKind::FusedAttentionSequence;
  }
  return PipelineStageMaterializationKind::SingleStage;
}

std::string_view fused_input_kind_to_string(PipelineFusedInputPlan::Kind kind) noexcept {
  switch (kind) {
  case PipelineFusedInputPlan::Kind::Output:
    return "output";
  case PipelineFusedInputPlan::Kind::External:
    return "external";
  case PipelineFusedInputPlan::Kind::None:
  default:
    return "none";
  }
}

PipelineFusedInputPlan::Kind fused_input_kind_from_string(
    std::string_view value) noexcept {
  if (value == "output") {
    return PipelineFusedInputPlan::Kind::Output;
  }
  if (value == "external") {
    return PipelineFusedInputPlan::Kind::External;
  }
  return PipelineFusedInputPlan::Kind::None;
}

void append_launch_plan(std::ostringstream &os,
                        const KernelLaunchPlanDescriptor &plan) {
  append_bool(os, plan.valid);
  append_string_vector(os, plan.buffer_roles);
  append_integral_vector(os, plan.direct_input_indices);
  append_integral_vector(os, plan.input_indices);
  append_number(os, plan.input_arg_count);
  append_integral_vector(os, plan.operand_kinds);
  append_integral_vector(os, plan.operand_arg_indices);
  append_integral_vector(os, plan.scalar_args);
  append_integral_vector(os, plan.scalar_arg_kinds);
}

KernelLaunchPlanDescriptor read_launch_plan(WireReader &reader) {
  KernelLaunchPlanDescriptor plan;
  plan.valid = reader.bool_field("launch plan valid");
  plan.buffer_roles = reader.string_vector("launch plan buffer roles");
  plan.direct_input_indices =
      reader.size_vector("launch plan direct input indices");
  plan.input_indices = reader.size_vector("launch plan input indices");
  plan.input_arg_count = reader.size_field("launch plan input arg count");
  plan.operand_kinds = reader.i32_vector("launch plan operand kinds");
  plan.operand_arg_indices =
      reader.i32_vector("launch plan operand arg indices");
  plan.scalar_args = reader.i32_vector("launch plan scalar args");
  plan.scalar_arg_kinds = reader.u32_vector("launch plan scalar arg kinds");
  return plan;
}

void append_runtime_binding(std::ostringstream &os,
                            const RuntimeTensorBindingContract &binding) {
  append_field(os, binding.logical_name);
  append_field(os, binding.memory_region_id);
  append_field(os, binding.role);
  append_field(os, binding.element_type);
  append_field(os, binding.partial_shape);
  append_field(os, binding.layout);
  append_field(os, binding.storage_kind);
  append_field(os, binding.lifetime_class);
  append_field(os, binding.alias_group);
  append_field(os, binding.stateful_prebind_variable_id);
  append_field(os, binding.stateful_prebind_shape_rule);
  append_number(os, binding.stateful_prebind_shape_axis);
  append_bool(os, binding.external_binding);
  append_bool(os, binding.host_visible);
}

RuntimeTensorBindingContract read_runtime_binding(WireReader &reader) {
  RuntimeTensorBindingContract binding;
  binding.logical_name = reader.string_field("runtime binding logical name");
  binding.memory_region_id = reader.string_field("runtime binding memory region");
  binding.role = reader.string_field("runtime binding role");
  binding.element_type = reader.string_field("runtime binding element type");
  binding.partial_shape = reader.string_field("runtime binding shape");
  binding.layout = reader.string_field("runtime binding layout");
  binding.storage_kind = reader.string_field("runtime binding storage");
  binding.lifetime_class = reader.string_field("runtime binding lifetime");
  binding.alias_group = reader.string_field("runtime binding alias group");
  binding.stateful_prebind_variable_id =
      reader.string_field("runtime binding stateful variable");
  binding.stateful_prebind_shape_rule =
      reader.string_field("runtime binding stateful shape rule");
  binding.stateful_prebind_shape_axis =
      reader.i64_field("runtime binding stateful shape axis");
  binding.external_binding = reader.bool_field("runtime binding external");
  binding.host_visible = reader.bool_field("runtime binding host visible");
  return binding;
}

void append_runtime_stage_descriptor(
    std::ostringstream &os, const RuntimeStageExecutableDescriptor &descriptor) {
  append_number(os, descriptor.stage_index);
  append_number(os, descriptor.stage_record_key);
  append_number(os, descriptor.artifact_descriptor_index);
  append_field(os, descriptor.manifest_ref);
  append_field(os, descriptor.abi_fingerprint);
  append_field(os, descriptor.artifact_key);
  append_field(os, descriptor.backend_domain);
  append_field(os, descriptor.kernel_id);
  append_field(os, descriptor.op_family);
  append_field(os, descriptor.stage_name);
  append_field(os, origin_to_string(descriptor.origin));
  append_field(os, payload_kind_to_string(descriptor.payload_kind));
  append_field(os, descriptor.entry_point);
  append_field(os, descriptor.compile_options_key);
  append_number(os, descriptor.abi_arg_count);
  append_number(os, descriptor.abi_output_arg_count);
  append_field(os, descriptor.dispatch_contract);
  append_field(os, descriptor.layout_contract);
  append_field(os, descriptor.runtime_shape_rule);
  append_integral_vector(os, descriptor.runtime_shape_i64_metadata);
  append_bool(os, descriptor.requires_runtime_shape_args);
  append_bool(os, descriptor.tensor_view_only);
  append_number(os, descriptor.submission_stage_weight);
  append_number(os, descriptor.submission_macs_estimate);
  append_bool(os, descriptor.submission_dependency_boundary);
  append_field(os, descriptor.stateful_effect);
  append_field(os, descriptor.stateful_variable_id);
  append_string_vector(os, descriptor.tensor_roles);
  append_string_vector(os, descriptor.scalar_roles);
  append_number(os, descriptor.runtime_param_buffer_count);
  append_field(os, runtime_param_descriptor_payload_kind_to_string(
                       descriptor.runtime_param_payload_kind));
  append_integral_vector(os, descriptor.runtime_param_i64_metadata);
  append_bool(os, descriptor.runtime_param_reduce_keep_dims);
  append_bool(os, descriptor.runtime_param_reduce_keep_dims_valid);
  append_launch_plan(os, descriptor.launch_plan);
  append_field(os, descriptor.exception_ticket);
  append_field(os, descriptor.exception_reason);
  append_field(os, descriptor.exception_removal_condition);
  append_bool(os, descriptor.optional_cache_payload_allowed);
  append_number(os, descriptor.input_bindings.size());
  for (const auto &binding : descriptor.input_bindings) {
    append_runtime_binding(os, binding);
  }
  append_number(os, descriptor.output_bindings.size());
  for (const auto &binding : descriptor.output_bindings) {
    append_runtime_binding(os, binding);
  }
}

RuntimeStageExecutableDescriptor read_runtime_stage_descriptor(WireReader &reader) {
  RuntimeStageExecutableDescriptor descriptor;
  descriptor.stage_index = reader.size_field("runtime stage index");
  descriptor.stage_record_key = reader.u64_field("runtime stage key");
  descriptor.artifact_descriptor_index =
      reader.size_field("runtime stage artifact index");
  descriptor.manifest_ref = reader.string_field("runtime stage manifest ref");
  descriptor.abi_fingerprint = reader.string_field("runtime stage abi");
  descriptor.artifact_key = reader.string_field("runtime stage artifact key");
  descriptor.backend_domain = reader.string_field("runtime stage backend");
  descriptor.kernel_id = reader.string_field("runtime stage kernel");
  descriptor.op_family = reader.string_field("runtime stage op family");
  descriptor.stage_name = reader.string_field("runtime stage name");
  descriptor.origin = origin_from_string(reader.string_field("runtime stage origin"));
  descriptor.payload_kind =
      payload_kind_from_string(reader.string_field("runtime stage payload kind"));
  descriptor.entry_point = reader.string_field("runtime stage entry point");
  descriptor.compile_options_key =
      reader.string_field("runtime stage compile options");
  descriptor.abi_arg_count = reader.u32_field("runtime stage abi arg count");
  descriptor.abi_output_arg_count =
      reader.u32_field("runtime stage abi output count");
  descriptor.dispatch_contract = reader.string_field("runtime stage dispatch");
  descriptor.layout_contract = reader.string_field("runtime stage layout");
  descriptor.runtime_shape_rule =
      reader.string_field("runtime stage shape rule");
  descriptor.runtime_shape_i64_metadata =
      reader.i64_vector("runtime stage shape metadata");
  descriptor.requires_runtime_shape_args =
      reader.bool_field("runtime stage requires shape args");
  descriptor.tensor_view_only = reader.bool_field("runtime stage view only");
  descriptor.submission_stage_weight =
      reader.u32_field("runtime stage submission weight");
  descriptor.submission_macs_estimate =
      reader.u64_field("runtime stage submission macs");
  descriptor.submission_dependency_boundary =
      reader.bool_field("runtime stage submission boundary");
  descriptor.stateful_effect = reader.string_field("runtime stage stateful effect");
  descriptor.stateful_variable_id =
      reader.string_field("runtime stage stateful variable");
  descriptor.tensor_roles = reader.string_vector("runtime stage tensor roles");
  descriptor.scalar_roles = reader.string_vector("runtime stage scalar roles");
  descriptor.runtime_param_buffer_count =
      reader.u32_field("runtime stage runtime param buffers");
  descriptor.runtime_param_payload_kind =
      runtime_param_descriptor_payload_kind_from_string(
          reader.string_field("runtime stage runtime param payload kind"));
  descriptor.runtime_param_i64_metadata =
      reader.i64_vector("runtime stage runtime param metadata");
  descriptor.runtime_param_reduce_keep_dims =
      reader.bool_field("runtime stage reduce keep dims");
  descriptor.runtime_param_reduce_keep_dims_valid =
      reader.bool_field("runtime stage reduce keep dims valid");
  descriptor.launch_plan = read_launch_plan(reader);
  descriptor.exception_ticket = reader.string_field("runtime stage exception ticket");
  descriptor.exception_reason = reader.string_field("runtime stage exception reason");
  descriptor.exception_removal_condition =
      reader.string_field("runtime stage exception removal");
  descriptor.optional_cache_payload_allowed =
      reader.bool_field("runtime stage optional cache payload");
  const auto input_count = reader.size_field("runtime stage input binding count");
  descriptor.input_bindings.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    descriptor.input_bindings.push_back(read_runtime_binding(reader));
  }
  const auto output_count = reader.size_field("runtime stage output binding count");
  descriptor.output_bindings.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    descriptor.output_bindings.push_back(read_runtime_binding(reader));
  }
  return descriptor;
}

void append_tensor_ref(std::ostringstream &os, const PipelineStageTensorRef &ref) {
  append_field(os, tensor_ref_kind_to_string(ref.kind));
  append_number(os, ref.index);
  append_number(os, ref.port);
}

PipelineStageTensorRef read_tensor_ref(WireReader &reader) {
  PipelineStageTensorRef ref;
  ref.kind = tensor_ref_kind_from_string(reader.string_field("tensor ref kind"));
  ref.index = reader.size_field("tensor ref index");
  ref.port = reader.size_field("tensor ref port");
  return ref;
}

void append_io_plan(std::ostringstream &os, const PipelineStageIoPlan &plan) {
  append_field(os, plan.stage_name);
  append_field(os, plan.op_family);
  append_number(os, plan.runtime_stage_index);
  append_number(os, plan.inputs.size());
  for (const auto &input : plan.inputs) {
    append_number(os, input.port);
    append_tensor_ref(os, input.source_ref);
  }
  append_number(os, plan.outputs.size());
  for (const auto &output : plan.outputs) {
    append_field(os, shape_to_contract(output.shape));
    append_field(os, output.type.get_type_name());
    append_bool(os, output.is_model_output);
    append_number(os, output.source_port);
    append_field(os, output.direct_stateful_assign_variable_id);
    append_tensor_ref(os, output.source_ref);
  }
  append_number(os, plan.output_aliases.size());
  for (const auto &alias : plan.output_aliases) {
    append_number(os, alias.source_port);
    append_number(os, alias.output_port);
    append_tensor_ref(os, alias.source_ref);
  }
}

PipelineStageIoPlan read_io_plan(WireReader &reader) {
  PipelineStageIoPlan plan;
  plan.stage_name = reader.string_field("io stage name");
  plan.op_family = reader.string_field("io op family");
  plan.runtime_stage_index = reader.size_field("io runtime stage index");
  const auto input_count = reader.size_field("io input count");
  plan.inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    PipelineStageInputLink input;
    input.port = reader.size_field("io input port");
    input.source_ref = read_tensor_ref(reader);
    plan.inputs.push_back(input);
  }
  const auto output_count = reader.size_field("io output count");
  plan.outputs.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    PipelineStageOutputDesc output;
    output.shape = read_shape(reader, "io output shape");
    output.type = read_element_type(reader, "io output type");
    output.is_model_output = reader.bool_field("io output is model output");
    output.source_port = reader.size_field("io output source port");
    output.direct_stateful_assign_variable_id =
        reader.string_field("io output stateful assign");
    output.source_ref = read_tensor_ref(reader);
    plan.outputs.push_back(std::move(output));
  }
  const auto alias_count = reader.size_field("io output alias count");
  plan.output_aliases.reserve(alias_count);
  for (size_t i = 0; i < alias_count; ++i) {
    PipelineStageOutputAlias alias;
    alias.source_port = reader.size_field("io alias source port");
    alias.output_port = reader.size_field("io alias output port");
    alias.source_ref = read_tensor_ref(reader);
    plan.output_aliases.push_back(alias);
  }
  return plan;
}

void append_vendor_attention(std::ostringstream &os,
                             const PipelineVendorAttentionStagePlan &plan) {
  append_field(os, plan.name);
  append_runtime_stage_descriptor(os, plan.descriptor);
}

PipelineVendorAttentionStagePlan read_vendor_attention(WireReader &reader) {
  PipelineVendorAttentionStagePlan plan;
  plan.name = reader.string_field("vendor attention name");
  plan.descriptor = read_runtime_stage_descriptor(reader);
  return plan;
}

void append_fused_input(std::ostringstream &os, const PipelineFusedInputPlan &input) {
  append_field(os, fused_input_kind_to_string(input.kind));
  append_number(os, input.index);
}

PipelineFusedInputPlan read_fused_input(WireReader &reader) {
  PipelineFusedInputPlan input;
  input.kind = fused_input_kind_from_string(reader.string_field("fused input kind"));
  input.index = reader.size_field("fused input index");
  return input;
}

void append_post_ops(std::ostringstream &os,
                     const PipelineStagePostOpFusionPlan &post_ops) {
  append_bool(os, post_ops.input_activation.has_value());
  if (post_ops.input_activation) {
    append_number(os, post_ops.input_activation->input_idx);
    append_number(os, static_cast<int>(post_ops.input_activation->kind));
    append_number(os, post_ops.input_activation->alpha);
  }
  append_bool(os, post_ops.batchnorm.has_value());
  if (post_ops.batchnorm) {
    append_float_vector(os, post_ops.batchnorm->gamma);
    append_float_vector(os, post_ops.batchnorm->beta);
    append_float_vector(os, post_ops.batchnorm->mean);
    append_float_vector(os, post_ops.batchnorm->var);
    append_number(os, post_ops.batchnorm->epsilon);
  }
  append_bool(os, post_ops.bias.has_value());
  if (post_ops.bias) {
    append_float_vector(os, post_ops.bias->values);
    append_integral_vector(os, post_ops.bias->shape);
    append_field(os, post_ops.bias->element_type.get_type_name());
  }
  append_bool(os, post_ops.activation.has_value());
  if (post_ops.activation) {
    append_number(os, static_cast<int>(*post_ops.activation));
    append_number(os, post_ops.activation_alpha);
  }
}

PipelineStagePostOpFusionPlan read_post_ops(WireReader &reader) {
  PipelineStagePostOpFusionPlan post_ops;
  if (reader.bool_field("post op input activation present")) {
    PipelineStageInputActivationFusionPlan input_activation;
    input_activation.input_idx = reader.size_field("post op input activation index");
    input_activation.kind =
        static_cast<ActivationKind>(reader.i32_field("post op input activation kind"));
    input_activation.alpha = reader.float_field("post op input activation alpha");
    post_ops.input_activation = input_activation;
  }
  if (reader.bool_field("post op batchnorm present")) {
    BatchNormParams batchnorm;
    batchnorm.gamma = reader.float_vector("post op batchnorm gamma");
    batchnorm.beta = reader.float_vector("post op batchnorm beta");
    batchnorm.mean = reader.float_vector("post op batchnorm mean");
    batchnorm.var = reader.float_vector("post op batchnorm var");
    batchnorm.epsilon = reader.float_field("post op batchnorm epsilon");
    post_ops.batchnorm = std::move(batchnorm);
  }
  if (reader.bool_field("post op bias present")) {
    BiasParams bias;
    bias.values = reader.float_vector("post op bias values");
    bias.shape = reader.i64_vector("post op bias shape");
    bias.element_type = read_element_type(reader, "post op bias element type");
    post_ops.bias = std::move(bias);
  }
  if (reader.bool_field("post op activation present")) {
    post_ops.activation =
        static_cast<ActivationKind>(reader.i32_field("post op activation kind"));
    post_ops.activation_alpha = reader.float_field("post op activation alpha");
  }
  return post_ops;
}

void append_parallelism_band(std::ostringstream &os,
                             const GpuChunkDispatchBand &band) {
  append_number(os, band.min_work_per_elem);
  append_number(os, band.elems_per_dispatch);
  append_number(os, band.max_elems_per_dispatch);
  append_number(os, band.target_dispatches);
}

void append_parallelism_profile(std::ostringstream &os,
                                const GpuParallelismProfile &profile) {
  append_field(os, profile.profile_key);
  append_number(os, profile.preferred_simd_width);
  append_number(os, profile.subgroup_size);
  append_number(os, profile.max_total_threads_per_group);
  append_number(os, profile.max_threads_per_group[0]);
  append_number(os, profile.max_threads_per_group[1]);
  append_number(os, profile.max_threads_per_group[2]);
  append_bool(os, profile.supports_conv_output_channel_blocking);
  append_bool(os, profile.supports_conv_channel_block_spatial_tiling);
  append_bool(os, profile.sort_matmul_tiles_by_shape);
  append_bool(os, profile.enable_skinny_matmul_tiles);
  append_bool(os, profile.scale_conv_threads_for_large_spatial);
  append_bool(os, profile.scale_conv_threads_for_dense_reduction);
  append_bool(os, profile.scale_conv_threads_for_pointwise_reduction);
  append_bool(os, profile.conv_spatial_micro_tile_requires_large_output_area);
  append_number(os, profile.chunk_dispatch.small_total_elems_threshold);
  append_number(os, profile.chunk_dispatch.small_min_elems_per_dispatch);
  append_parallelism_band(os, profile.chunk_dispatch.light);
  append_parallelism_band(os, profile.chunk_dispatch.medium);
  append_parallelism_band(os, profile.chunk_dispatch.heavy);
  append_parallelism_band(os, profile.chunk_dispatch.very_heavy);
  append_bool(os, profile.chunk_dispatch.retune_threads_to_workload);
}

void append_materialization_plan(std::ostringstream &os,
                                 const PipelineStageMaterializationPlan &plan) {
  append_field(os, materialization_kind_to_string(plan.kind));
  append_io_plan(os, plan.io_plan);
  append_number(os, plan.descriptor_stage_index);
  append_runtime_stage_descriptor(os, plan.materialized_descriptor);
  append_bool(os, plan.materialized_descriptor_valid);
  append_vendor_attention(os, plan.vendor_attention);
  append_integral_vector(os, plan.fused_node_indices);
  append_integral_vector(os, plan.fused_descriptor_stage_indices);
  append_number(os, plan.fused_inner_stages.size());
  for (const auto &inner_stage : plan.fused_inner_stages) {
    append_integral_vector(os, inner_stage.output_indices);
    append_number(os, inner_stage.inputs.size());
    for (const auto &input : inner_stage.inputs) {
      append_fused_input(os, input);
    }
  }
  append_number(os, plan.input_transforms.size());
  for (const auto &binding : plan.input_transforms) {
    append_number(os, binding.input_idx);
    append_field(os, shape_to_contract(binding.transform.source_shape));
    append_integral_vector(os, binding.transform.transpose_permutation);
  }
  append_bool(os, plan.residual_add.has_value());
  append_post_ops(os, plan.post_ops);
}

PipelineStageMaterializationPlan read_materialization_plan(WireReader &reader) {
  PipelineStageMaterializationPlan plan;
  plan.kind = materialization_kind_from_string(
      reader.string_field("materialization kind"));
  plan.io_plan = read_io_plan(reader);
  plan.descriptor_stage_index =
      reader.size_field("materialization descriptor stage index");
  plan.materialized_descriptor = read_runtime_stage_descriptor(reader);
  plan.materialized_descriptor_valid =
      reader.bool_field("materialized descriptor valid");
  plan.vendor_attention = read_vendor_attention(reader);
  plan.fused_node_indices = reader.size_vector("materialization fused node indices");
  plan.fused_descriptor_stage_indices =
      reader.size_vector("materialization fused descriptor indices");
  const auto inner_stage_count =
      reader.size_field("materialization fused inner stage count");
  plan.fused_inner_stages.reserve(inner_stage_count);
  for (size_t i = 0; i < inner_stage_count; ++i) {
    PipelineFusedInnerStagePlan inner_stage;
    inner_stage.output_indices =
        reader.size_vector("materialization fused output indices");
    const auto input_count =
        reader.size_field("materialization fused input count");
    inner_stage.inputs.reserve(input_count);
    for (size_t input_idx = 0; input_idx < input_count; ++input_idx) {
      inner_stage.inputs.push_back(read_fused_input(reader));
    }
    plan.fused_inner_stages.push_back(std::move(inner_stage));
  }
  const auto transform_count =
      reader.size_field("materialization input transform count");
  plan.input_transforms.reserve(transform_count);
  for (size_t i = 0; i < transform_count; ++i) {
    PipelineStageInputTransformBinding binding;
    binding.input_idx = reader.size_field("materialization transform input");
    binding.transform.source_shape =
        read_shape(reader, "materialization transform source shape");
    binding.transform.transpose_permutation =
        reader.i64_vector("materialization transform permutation");
    plan.input_transforms.push_back(std::move(binding));
  }
  if (reader.bool_field("materialization residual add present")) {
    plan.residual_add = PipelineStageResidualAddFusionPlan{};
  }
  plan.post_ops = read_post_ops(reader);
  return plan;
}

void append_public_output(std::ostringstream &os,
                          const RuntimePublicOutputDescriptor &output) {
  append_field(os, public_output_kind_to_string(output.kind));
  append_number(os, output.index);
  append_number(os, output.port);
  append_field(os, shape_to_contract(output.static_shape));
  append_field(os, output.static_type.get_type_name());
}

RuntimePublicOutputDescriptor read_public_output(WireReader &reader) {
  RuntimePublicOutputDescriptor output;
  output.kind =
      public_output_kind_from_string(reader.string_field("public output kind"));
  output.index = reader.size_field("public output index");
  output.port = reader.size_field("public output port");
  output.static_shape = read_shape(reader, "public output shape");
  output.static_type = read_element_type(reader, "public output type");
  return output;
}

void append_runtime_options(std::ostringstream &os,
                            const PipelineStageRuntimeOptionsPlan &options) {
  append_bool(os, options.custom_kernel_dispatch_enabled);
  append_parallelism_profile(os, options.custom_kernel_dispatch_profile);
}

PipelineStageRuntimeOptionsPlan read_runtime_options(WireReader &reader) {
  PipelineStageRuntimeOptionsPlan options;
  options.custom_kernel_dispatch_enabled =
      reader.bool_field("runtime options custom dispatch enabled");
  auto &profile = options.custom_kernel_dispatch_profile;
  profile.profile_key = reader.string_field("parallelism profile key");
  profile.preferred_simd_width =
      reader.u32_field("parallelism preferred simd width");
  profile.subgroup_size = reader.u32_field("parallelism subgroup size");
  profile.max_total_threads_per_group =
      reader.u32_field("parallelism max total threads");
  profile.max_threads_per_group[0] = reader.u32_field("parallelism max x");
  profile.max_threads_per_group[1] = reader.u32_field("parallelism max y");
  profile.max_threads_per_group[2] = reader.u32_field("parallelism max z");
  profile.supports_conv_output_channel_blocking =
      reader.bool_field("parallelism conv oc blocking");
  profile.supports_conv_channel_block_spatial_tiling =
      reader.bool_field("parallelism conv channel block spatial tiling");
  profile.sort_matmul_tiles_by_shape =
      reader.bool_field("parallelism sort matmul tiles");
  profile.enable_skinny_matmul_tiles =
      reader.bool_field("parallelism skinny matmul tiles");
  profile.scale_conv_threads_for_large_spatial =
      reader.bool_field("parallelism scale conv large spatial");
  profile.scale_conv_threads_for_dense_reduction =
      reader.bool_field("parallelism scale conv dense reduction");
  profile.scale_conv_threads_for_pointwise_reduction =
      reader.bool_field("parallelism scale conv pointwise reduction");
  profile.conv_spatial_micro_tile_requires_large_output_area =
      reader.bool_field("parallelism conv micro tile large output area");
  profile.chunk_dispatch.small_total_elems_threshold =
      reader.size_field("parallelism chunk small threshold");
  profile.chunk_dispatch.small_min_elems_per_dispatch =
      reader.size_field("parallelism chunk small min elems");
  const auto read_band = [&](GpuChunkDispatchBand &band,
                             std::string_view label) {
    band.min_work_per_elem =
        reader.size_field(std::string(label) + " min work per elem");
    band.elems_per_dispatch =
        reader.size_field(std::string(label) + " elems per dispatch");
    band.max_elems_per_dispatch =
        reader.size_field(std::string(label) + " max elems per dispatch");
    band.target_dispatches =
        reader.size_field(std::string(label) + " target dispatches");
  };
  read_band(profile.chunk_dispatch.light, "parallelism light band");
  read_band(profile.chunk_dispatch.medium, "parallelism medium band");
  read_band(profile.chunk_dispatch.heavy, "parallelism heavy band");
  read_band(profile.chunk_dispatch.very_heavy, "parallelism very heavy band");
  profile.chunk_dispatch.retune_threads_to_workload =
      reader.bool_field("parallelism retune threads");
  return options;
}

const RuntimeStageExecutableDescriptor *find_seed_stage(
    const RuntimeExecutableDescriptor &descriptor,
    const RuntimeStageExecutableDescriptor &stage) {
  if (stage.stage_index < descriptor.stages.size()) {
    const auto &candidate = descriptor.stages[stage.stage_index];
    if (candidate.stage_record_key == stage.stage_record_key &&
        candidate.artifact_key == stage.artifact_key) {
      return &candidate;
    }
  }
  for (const auto &candidate : descriptor.stages) {
    if (candidate.stage_record_key == stage.stage_record_key &&
        candidate.artifact_key == stage.artifact_key) {
      return &candidate;
    }
  }
  return nullptr;
}

void rebind_stage_payload(RuntimeStageExecutableDescriptor &stage,
                          const RuntimeExecutableDescriptor &descriptor) {
  if (const auto *seed = find_seed_stage(descriptor, stage)) {
    stage.payload = seed->payload;
    stage.const_tensors = seed->const_tensors;
  }
}

void rebind_materialization_payloads(RuntimeExecutableDescriptor &descriptor) {
  for (auto &plan : descriptor.materialization_stages) {
    rebind_stage_payload(plan.materialized_descriptor, descriptor);
    rebind_stage_payload(plan.vendor_attention.descriptor, descriptor);
  }
}

} // namespace

CacheMaterializationContract make_cache_materialization_contract(
    const RuntimeExecutableDescriptor &descriptor) {
  CacheMaterializationContract contract;
  contract.finalized = descriptor.materialization_finalized;
  contract.stages = descriptor.materialization_stages;
  contract.public_outputs = descriptor.public_outputs;
  contract.runtime_options = descriptor.runtime_options;
  return contract;
}

std::string serialize_cache_materialization_contract(
    const CacheMaterializationContract &contract) {
  std::ostringstream os;
  append_field(os, "GFX_CACHE_MATERIALIZATION");
  append_field(os, "2");
  append_bool(os, contract.finalized);
  append_number(os, contract.stages.size());
  for (const auto &stage : contract.stages) {
    append_materialization_plan(os, stage);
  }
  append_number(os, contract.public_outputs.size());
  for (const auto &output : contract.public_outputs) {
    append_public_output(os, output);
  }
  append_runtime_options(os, contract.runtime_options);
  return os.str();
}

CacheMaterializationContract deserialize_cache_materialization_contract(
    std::string_view wire, std::vector<std::string> &diagnostics) {
  CacheMaterializationContract contract;
  WireReader reader(wire);
  const auto magic = reader.string_field("cache materialization magic");
  const auto version = reader.string_field("cache materialization version");
  if (magic != "GFX_CACHE_MATERIALIZATION") {
    diagnostics.emplace_back("cache materialization wire magic mismatch");
  }
  if (version != "2") {
    diagnostics.emplace_back("cache materialization wire version mismatch");
  }
  contract.finalized = reader.bool_field("cache materialization finalized");
  const auto stage_count = reader.size_field("cache materialization stage count");
  contract.stages.reserve(stage_count);
  for (size_t i = 0; i < stage_count; ++i) {
    contract.stages.push_back(read_materialization_plan(reader));
  }
  const auto output_count =
      reader.size_field("cache materialization public output count");
  contract.public_outputs.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    contract.public_outputs.push_back(read_public_output(reader));
  }
  contract.runtime_options = read_runtime_options(reader);
  auto read_diagnostics = reader.take_diagnostics();
  diagnostics.insert(diagnostics.end(), read_diagnostics.begin(),
                     read_diagnostics.end());
  return contract;
}

RuntimeExecutableDescriptor apply_cache_materialization_contract(
    const CacheMaterializationContract &contract,
    RuntimeExecutableDescriptor descriptor) {
  descriptor.materialization_finalized = contract.finalized;
  descriptor.materialization_stages = contract.stages;
  descriptor.public_outputs = contract.public_outputs;
  descriptor.runtime_options = contract.runtime_options;
  rebind_materialization_payloads(descriptor);
  return descriptor;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
