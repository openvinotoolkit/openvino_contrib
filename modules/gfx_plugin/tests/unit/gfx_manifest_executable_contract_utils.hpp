// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "unit/gfx_backend_architecture_contract_common.hpp"

namespace ov {
namespace gfx_plugin {
namespace {


compiler::TensorContract
make_tensor_contract(compiler::TensorContractRole role) {
  compiler::TensorContract contract;
  contract.logical_name =
      role == compiler::TensorContractRole::TensorInput ? "input0" : "output0";
  contract.memory_region_id = role == compiler::TensorContractRole::TensorInput
                                  ? "stage_0.input_0"
                                  : "stage_0.output_0";
  contract.role = role;
  contract.element_type = "f32";
  contract.partial_shape = "{1,3}";
  contract.layout = "logical";
  contract.storage_kind = "device_buffer";
  contract.lifetime_class = role == compiler::TensorContractRole::TensorInput
                                ? "producer_or_external"
                                : "stage_output";
  return contract;
}

compiler::MemoryRegion
make_memory_region_for_contract(const compiler::TensorContract &contract,
                                size_t stage_id) {
  compiler::MemoryRegion region;
  region.region_id = contract.memory_region_id;
  region.logical_tensor_name = contract.logical_name;
  region.kind = contract.role == compiler::TensorContractRole::TensorInput
                    ? compiler::MemoryRegionKind::ExternalTensor
                    : compiler::MemoryRegionKind::TransientTensor;
  region.element_type = contract.element_type;
  region.partial_shape = contract.partial_shape;
  region.layout = contract.layout;
  region.storage_kind = contract.storage_kind;
  region.alias_group =
      contract.role == compiler::TensorContractRole::TensorInput
          ? contract.memory_region_id
          : "stage_" + std::to_string(stage_id);
  region.lifetime = {0, stage_id};
  region.external_binding =
      contract.role == compiler::TensorContractRole::TensorInput;
  return region;
}

compiler::MemoryPlan
make_single_stage_memory_plan(const compiler::StageRecord &stage) {
  compiler::MemoryPlan plan;
  plan.schema_version = 1;
  for (const auto &input : stage.inputs) {
    auto region = make_memory_region_for_contract(input, stage.stage_id);
    compiler::AliasGroup group;
    group.group_id = region.alias_group;
    group.region_ids.push_back(region.region_id);
    plan.alias_groups.push_back(std::move(group));
    plan.regions.push_back(std::move(region));
  }
  compiler::TransientArena arena;
  arena.arena_id = "transient_device_buffer_arena";
  arena.storage_kind = "device_buffer";
  compiler::AliasGroup output_group;
  output_group.group_id = stage.memory.alias_group;
  for (const auto &output : stage.outputs) {
    auto region = make_memory_region_for_contract(output, stage.stage_id);
    output_group.region_ids.push_back(region.region_id);
    arena.region_ids.push_back(region.region_id);
    plan.regions.push_back(std::move(region));
  }
  if (!output_group.region_ids.empty()) {
    plan.alias_groups.push_back(std::move(output_group));
  }
  if (!arena.region_ids.empty()) {
    plan.transient_arenas.push_back(std::move(arena));
  }
  return plan;
}

compiler::ManifestBundle make_single_payload_route_manifest(
    LoweringRouteKind route_kind, std::string backend_domain,
    std::string kernel_unit_id, std::string kernel_unit_kind,
    std::string op_family = "PayloadRoute",
    bool requires_runtime_shape_args = false) {
  compiler::StageRecord stage;
  stage.stage_id = 0;
  stage.stable_record_key = 0x1234u;
  stage.source_node_name = op_family;
  stage.normalized_op_family = std::move(op_family);
  stage.execution_kind = route_kind;
  stage.backend_domain = std::move(backend_domain);
  stage.kernel_unit_id = std::move(kernel_unit_id);
  stage.kernel_unit_kind = std::move(kernel_unit_kind);
  stage.requires_runtime_shape_args = requires_runtime_shape_args;
  stage.inputs.push_back(
      make_tensor_contract(compiler::TensorContractRole::TensorInput));
  stage.outputs.push_back(
      make_tensor_contract(compiler::TensorContractRole::TensorOutput));
  stage.dispatch.execution_kind = stage.execution_kind;
  stage.dispatch.backend_domain = stage.backend_domain;
  stage.dispatch.kernel_unit_id = stage.kernel_unit_id;
  stage.dispatch.kernel_unit_kind = stage.kernel_unit_kind;
  stage.dispatch.dispatch_source = "manifest";
  stage.memory.alias_group = "stage_0";

  compiler::ManifestBundle manifest;
  manifest.schema_version = 2;
  manifest.target_fingerprint = stage.backend_domain + ":test-target";
  manifest.memory_plan = make_single_stage_memory_plan(stage);
  manifest.stages.push_back(std::move(stage));
  return manifest;
}

compiler::TensorContract
make_source_payload_tensor_contract(compiler::TensorContractRole role,
                                    size_t index, std::string shape) {
  auto contract = make_tensor_contract(role);
  const bool input = role == compiler::TensorContractRole::TensorInput;
  contract.logical_name = (input ? "input" : "output") + std::to_string(index);
  contract.memory_region_id = "stage_0." + contract.logical_name + "_region";
  contract.partial_shape = std::move(shape);
  return contract;
}

compiler::ManifestBundle make_source_payload_route_manifest(
    std::string backend_domain, std::string op_family,
    const std::vector<std::string> &input_shapes,
    const std::vector<std::string> &output_shapes) {
  compiler::StageRecord stage;
  stage.stage_id = 0;
  stage.stable_record_key = 0x1234u;
  stage.source_node_name = op_family;
  stage.normalized_op_family = std::move(op_family);
  stage.execution_kind = LoweringRouteKind::GeneratedKernel;
  stage.backend_domain = std::move(backend_domain);
  stage.kernel_unit_id = stage.backend_domain + "/generated/unit_source";
  stage.kernel_unit_kind = "generated_kernel";
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    stage.inputs.push_back(make_source_payload_tensor_contract(
        compiler::TensorContractRole::TensorInput, i, input_shapes[i]));
  }
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    stage.outputs.push_back(make_source_payload_tensor_contract(
        compiler::TensorContractRole::TensorOutput, i, output_shapes[i]));
  }
  stage.dispatch.execution_kind = stage.execution_kind;
  stage.dispatch.backend_domain = stage.backend_domain;
  stage.dispatch.kernel_unit_id = stage.kernel_unit_id;
  stage.dispatch.kernel_unit_kind = stage.kernel_unit_kind;
  stage.dispatch.dispatch_source = "manifest";
  stage.memory.alias_group = "stage_0";

  compiler::ManifestBundle manifest;
  manifest.schema_version = 2;
  manifest.target_fingerprint = stage.backend_domain + ":test-target";
  manifest.memory_plan = make_single_stage_memory_plan(stage);
  manifest.stages.push_back(std::move(stage));
  return manifest;
}

GfxKernelStageManifest
make_unit_source_stage_manifest(GfxKernelBackendDomain backend_domain,
                                const std::vector<GfxKernelBufferRole> &roles) {
  GfxKernelStageManifest manifest;
  manifest.valid = true;
  manifest.stage_family = GfxKernelStageFamily::Eltwise;
  manifest.backend_domain = backend_domain;
  manifest.execution_kind = GfxKernelExecutionKind::CustomKernel;
  manifest.storage = GfxKernelStorageKind::Buffer;
  manifest.compute_precision = GfxKernelComputePrecision::Native;
  manifest.custom_kernel.valid = true;
  manifest.custom_kernel.kernel_family = "unit_source";
  manifest.custom_kernel.kernel_family_id = 1;
  manifest.custom_kernel.entry_point = "unit_source_entry";
  manifest.custom_kernel.external_buffer_abi = make_gfx_kernel_roles_abi(roles);
  return manifest;
}

uint32_t
count_runtime_param_roles(const std::vector<GfxKernelBufferRole> &roles) {
  uint32_t count = 0;
  for (const auto role : roles) {
    if (role == GfxKernelBufferRole::RuntimeParams) {
      ++count;
    }
  }
  return count;
}

uint32_t count_kernel_roles(const std::vector<GfxKernelBufferRole> &roles,
                            GfxKernelBufferRole expected) {
  uint32_t count = 0;
  for (const auto role : roles) {
    if (role == expected) {
      ++count;
    }
  }
  return count;
}

std::vector<size_t>
make_unit_direct_input_indices(const std::vector<GfxKernelBufferRole> &roles) {
  std::vector<size_t> indices;
  size_t next_input = 0;
  for (const auto role : roles) {
    if (role == GfxKernelBufferRole::TensorInput) {
      indices.push_back(next_input++);
    }
  }
  return indices;
}

KernelLaunchPlanDescriptor make_unit_launch_plan_descriptor(
    const std::vector<GfxKernelBufferRole> &roles,
    const GfxKernelSourceRuntimeBinding &runtime_binding) {
  KernelLaunchPlanDescriptor plan;
  plan.valid = !roles.empty();
  plan.buffer_roles.reserve(roles.size());
  for (const auto role : roles) {
    plan.buffer_roles.emplace_back(kernel_buffer_role_descriptor_name(role));
  }
  plan.direct_input_indices = make_unit_direct_input_indices(roles);
  plan.input_indices = runtime_binding.inputs;
  plan.input_arg_count = runtime_binding.input_arg_count;
  plan.operand_kinds = runtime_binding.operand_kinds;
  plan.operand_arg_indices = runtime_binding.operand_arg_indices;
  plan.scalar_args = runtime_binding.scalar_args;
  return plan;
}

std::shared_ptr<const KernelArtifactPayload> make_unit_source_payload(
    const compiler::KernelArtifactDescriptor &descriptor,
    const std::vector<GfxKernelBufferRole> &roles,
    const GfxKernelSourceRuntimeBinding &runtime_binding = {}) {
  if (descriptor.payload_kind == KernelArtifactPayloadKind::OpenClSource) {
    GfxOpenClSourceArtifact artifact;
    artifact.valid = true;
    artifact.artifact_ref.valid = true;
    artifact.artifact_ref.kind = GfxKernelArtifactKind::OpenClSource;
    artifact.artifact_ref.backend_domain = GfxKernelBackendDomain::OpenCl;
    artifact.artifact_ref.source_id = descriptor.kernel.kernel_id;
    artifact.artifact_ref.entry_point = descriptor.entry_point;
    artifact.source = "__kernel void unit_source_entry() {}";
    artifact.stage_manifest =
        make_unit_source_stage_manifest(GfxKernelBackendDomain::OpenCl, roles);
    artifact.arg_count = static_cast<uint32_t>(roles.size());
    artifact.direct_input_count =
        count_kernel_roles(roles, GfxKernelBufferRole::TensorInput);
    artifact.direct_output_count =
        count_kernel_roles(roles, GfxKernelBufferRole::TensorOutput);
    artifact.direct_input_indices = make_unit_direct_input_indices(roles);
    return std::make_shared<GfxOpenClSourceArtifactPayload>(
        std::move(artifact));
  }

  return std::make_shared<GfxKernelSourcePayload>(
      descriptor.kernel.kernel_id, descriptor.kernel.backend_domain,
      descriptor.entry_point, GfxKernelSourceLanguage::MetalShadingLanguage,
      "kernel void unit_source_entry() {}",
      make_unit_source_stage_manifest(GfxKernelBackendDomain::AppleMsl, roles),
      runtime_binding);
}

compiler::ExecutableBundle make_source_payload_executable(
    std::string backend_domain, std::string op_family,
    const std::vector<GfxKernelBufferRole> &roles,
    const std::vector<std::string> &input_shapes,
    const std::vector<std::string> &output_shapes,
    const GfxKernelSourceRuntimeBinding &runtime_binding = {},
    std::vector<KernelArtifactConstTensor> const_tensors = {}) {
  auto manifest = make_source_payload_route_manifest(
      std::move(backend_domain), std::move(op_family), input_shapes,
      output_shapes);
  auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  OPENVINO_ASSERT(executable.artifact_descriptors.size() == 1,
                  "unit source payload executable must have one descriptor");
  auto &descriptor = executable.artifact_descriptors.front();
  descriptor.entry_point = "unit_source_entry";
  descriptor.abi_arg_count = static_cast<uint32_t>(roles.size());
  descriptor.abi_output_arg_count =
      count_kernel_roles(roles, GfxKernelBufferRole::TensorOutput);
  descriptor.runtime_param_buffer_count = count_runtime_param_roles(roles);
  descriptor.runtime_param_payload_kind =
      runtime_param_descriptor_payload_kind_for_stage(
          descriptor.kernel.op_family, descriptor.runtime_param_buffer_count);
  descriptor.runtime_param_i64_metadata =
      runtime_binding.runtime_param_i64_metadata;
  descriptor.runtime_param_reduce_keep_dims =
      runtime_binding.runtime_param_reduce_keep_dims;
  descriptor.runtime_param_reduce_keep_dims_valid =
      runtime_binding.runtime_param_reduce_keep_dims_valid;
  descriptor.launch_plan =
      make_unit_launch_plan_descriptor(roles, runtime_binding);
  compiler::finalize_kernel_artifact_descriptor_identity(descriptor);
  compiler::KernelArtifactPayloadRecord payload_record;
  payload_record.artifact_descriptor_index = 0;
  payload_record.artifact_key = descriptor.artifact_key;
  payload_record.payload =
      make_unit_source_payload(descriptor, roles, runtime_binding);
  payload_record.const_tensors = std::move(const_tensors);
  executable.artifact_payloads.push_back(std::move(payload_record));
  return executable;
}

std::shared_ptr<ov::Model> make_add_constant_model(float value) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 3});
  auto constant = ov::op::v0::Constant::create(
      ov::element::f32, ov::Shape{1, 3}, {value, value, value});
  auto add = std::make_shared<ov::op::v1::Add>(input, constant);
  auto result = std::make_shared<ov::op::v0::Result>(add);
  return std::make_shared<ov::Model>(ov::ResultVector{result},
                                     ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> make_reshape_model(bool special_zero) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{2, 3});
  auto pattern =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
  auto reshape =
      std::make_shared<ov::op::v1::Reshape>(input, pattern, special_zero);
  auto result = std::make_shared<ov::op::v0::Result>(reshape);
  return std::make_shared<ov::Model>(ov::ResultVector{result},
                                     ov::ParameterVector{input});
}

compiler::CacheEnvelopeBuildOptions make_test_cache_options(
    const ov::Model &model,
    std::string backend_capabilities_fingerprint = "test-capabilities-v1",
    std::string backend_compiler_revision = "test-backend-compiler-v1",
    std::string driver_identity = "test-driver-v1") {
  compiler::CacheEnvelopeBuildOptions options;
  options.model_fingerprint = compiler::make_model_cache_fingerprint(model);
  options.backend_capabilities_fingerprint =
      std::move(backend_capabilities_fingerprint);
  options.backend_compiler_revision = std::move(backend_compiler_revision);
  options.driver_identity = std::move(driver_identity);
  return options;
}

compiler::PlannedOperation
make_metadata_planned_operation(const std::shared_ptr<const ov::Node> &node,
                                compiler::TensorLayoutPlan layout,
                                bool requires_runtime_shape_args = false) {
  compiler::PlannedOperation op;
  op.source_node = node;
  op.node_name = node ? node->get_friendly_name() : "metadata";
  op.type_name = node ? node->get_type_name() : "Unknown";
  op.kernel_unit = compiler::KernelUnit::describe(
      LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
      "opencl", "metadata", requires_runtime_shape_args);
  op.layout = layout;
  op.profitability_score = 1.0;
  op.input_element_types = {"f32"};
  op.input_shapes = {"{1,2,3}"};
  op.output_element_types = {"f32"};
  op.output_shapes = {"{1,2,3}"};
  return op;
}

RuntimeStageExecutableDescriptor
make_runtime_descriptor_for_layout(const std::shared_ptr<const ov::Node> &node,
                                   compiler::TensorLayoutPlan layout) {
  compiler::LoweringPlan plan;
  plan.target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  plan.operations.push_back(make_metadata_planned_operation(node, layout));
  const auto manifest = compiler::ManifestBuilder{}.build(plan);
  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  const auto runtime_descriptor =
      compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
  EXPECT_EQ(runtime_descriptor.stages.size(), 1u);
  return runtime_descriptor.stages.front();
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
