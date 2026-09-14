// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/cache_envelope.hpp"

#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "compiler/cache_materialization_contract.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

void append_field(std::ostringstream &os, std::string_view value) {
  os << value.size() << ":" << value << ";";
}

void append_bool(std::ostringstream &os, bool value) {
  append_field(os, value ? "1" : "0");
}

template <typename T>
void append_number(std::ostringstream &os, T value) {
  append_field(os, std::to_string(value));
}

void append_number(std::ostringstream &os, float value) {
  std::ostringstream value_os;
  value_os << std::setprecision(std::numeric_limits<float>::max_digits10)
           << value;
  append_field(os, value_os.str());
}

void append_number(std::ostringstream &os, double value) {
  std::ostringstream value_os;
  value_os << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
  append_field(os, value_os.str());
}

template <typename T>
void append_vector(std::ostringstream &os, const std::vector<T> &values) {
  append_field(os, std::to_string(values.size()));
  for (const auto &value : values) {
    append_number(os, value);
  }
}

void append_vector(std::ostringstream &os,
                   const std::vector<std::string> &values) {
  append_field(os, std::to_string(values.size()));
  for (const auto &value : values) {
    append_field(os, value);
  }
}

template <typename T>
void append_integral_vector(std::ostringstream &os,
                            const std::vector<T> &values) {
  append_field(os, std::to_string(values.size()));
  for (const auto value : values) {
    append_field(os, std::to_string(value));
  }
}

LoweringRouteKind lowering_route_kind_from_string(std::string_view value) {
  if (value == lowering_route_kind_to_string(LoweringRouteKind::Common)) {
    return LoweringRouteKind::Common;
  }
  if (value == lowering_route_kind_to_string(LoweringRouteKind::Metadata)) {
    return LoweringRouteKind::Metadata;
  }
  if (value ==
      lowering_route_kind_to_string(LoweringRouteKind::VendorPrimitive)) {
    return LoweringRouteKind::VendorPrimitive;
  }
  if (value ==
      lowering_route_kind_to_string(LoweringRouteKind::GeneratedKernel)) {
    return LoweringRouteKind::GeneratedKernel;
  }
  if (value == lowering_route_kind_to_string(
                   LoweringRouteKind::HandwrittenKernelException)) {
    return LoweringRouteKind::HandwrittenKernelException;
  }
  return LoweringRouteKind::Unsupported;
}

TensorContractRole tensor_contract_role_from_string(std::string_view value) {
  if (value == tensor_contract_role_to_string(TensorContractRole::TensorOutput)) {
    return TensorContractRole::TensorOutput;
  }
  return TensorContractRole::TensorInput;
}

RuntimeParamKind runtime_param_kind_from_string(std::string_view value) {
  if (value == runtime_param_kind_to_string(RuntimeParamKind::Scalar)) {
    return RuntimeParamKind::Scalar;
  }
  return RuntimeParamKind::Shape;
}

StatefulEffectKind stateful_effect_kind_from_string(std::string_view value) {
  if (value == stateful_effect_kind_to_string(StatefulEffectKind::ReadValue)) {
    return StatefulEffectKind::ReadValue;
  }
  if (value == stateful_effect_kind_to_string(StatefulEffectKind::Assign)) {
    return StatefulEffectKind::Assign;
  }
  return StatefulEffectKind::None;
}

MemoryRegionKind memory_region_kind_from_string(std::string_view value) {
  if (value == memory_region_kind_to_string(MemoryRegionKind::ExternalTensor)) {
    return MemoryRegionKind::ExternalTensor;
  }
  if (value == memory_region_kind_to_string(MemoryRegionKind::ImmutableTensor)) {
    return MemoryRegionKind::ImmutableTensor;
  }
  return MemoryRegionKind::TransientTensor;
}

KernelArtifactOrigin kernel_artifact_origin_from_string(
    std::string_view value) {
  if (value == kernel_artifact_origin_to_string(KernelArtifactOrigin::Common)) {
    return KernelArtifactOrigin::Common;
  }
  if (value == kernel_artifact_origin_to_string(KernelArtifactOrigin::Metadata)) {
    return KernelArtifactOrigin::Metadata;
  }
  if (value == kernel_artifact_origin_to_string(
                   KernelArtifactOrigin::VendorPrimitive)) {
    return KernelArtifactOrigin::VendorPrimitive;
  }
  if (value == kernel_artifact_origin_to_string(KernelArtifactOrigin::Generated)) {
    return KernelArtifactOrigin::Generated;
  }
  if (value == kernel_artifact_origin_to_string(
                   KernelArtifactOrigin::HandwrittenException)) {
    return KernelArtifactOrigin::HandwrittenException;
  }
  return KernelArtifactOrigin::Unknown;
}

KernelArtifactPayloadKind kernel_artifact_payload_kind_from_string(
    std::string_view value) {
  if (value == kernel_artifact_payload_kind_to_string(
                   KernelArtifactPayloadKind::VendorDescriptor)) {
    return KernelArtifactPayloadKind::VendorDescriptor;
  }
  if (value ==
      kernel_artifact_payload_kind_to_string(KernelArtifactPayloadKind::MslSource)) {
    return KernelArtifactPayloadKind::MslSource;
  }
  if (value == kernel_artifact_payload_kind_to_string(
                   KernelArtifactPayloadKind::OpenClSource)) {
    return KernelArtifactPayloadKind::OpenClSource;
  }
  return KernelArtifactPayloadKind::None;
}


class WireReader final {
public:
  explicit WireReader(std::string_view wire) : m_wire(wire) {}

  bool ok() const noexcept { return m_diagnostics.empty(); }
  std::vector<std::string> take_diagnostics() { return std::move(m_diagnostics); }

  std::string string_field(std::string_view name) {
    if (m_pos >= m_wire.size()) {
      m_diagnostics.push_back(std::string("cache envelope wire ended before ") +
                              std::string(name));
      return {};
    }
    size_t colon = m_wire.find(':', m_pos);
    if (colon == std::string_view::npos) {
      m_diagnostics.push_back(std::string("cache envelope wire field ") +
                              std::string(name) + " has no length separator");
      m_pos = m_wire.size();
      return {};
    }
    const auto length_text = m_wire.substr(m_pos, colon - m_pos);
    size_t length = 0;
    try {
      length = static_cast<size_t>(std::stoull(std::string(length_text)));
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("cache envelope wire field ") +
                              std::string(name) + " has invalid length");
      m_pos = m_wire.size();
      return {};
    }
    const size_t value_begin = colon + 1;
    const size_t value_end = value_begin + length;
    if (value_end >= m_wire.size() || m_wire[value_end] != ';') {
      m_diagnostics.push_back(std::string("cache envelope wire field ") +
                              std::string(name) + " is truncated");
      m_pos = m_wire.size();
      return {};
    }
    m_pos = value_end + 1;
    return std::string(m_wire.substr(value_begin, length));
  }

  uint32_t u32_field(std::string_view name) {
    return static_cast<uint32_t>(u64_field(name));
  }

  uint64_t u64_field(std::string_view name) {
    const auto value = string_field(name);
    try {
      return static_cast<uint64_t>(std::stoull(value));
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("cache envelope wire field ") +
                              std::string(name) + " is not uint64");
      return 0;
    }
  }

  size_t size_field(std::string_view name) {
    return static_cast<size_t>(u64_field(name));
  }

  int64_t i64_field(std::string_view name) {
    const auto value = string_field(name);
    try {
      return static_cast<int64_t>(std::stoll(value));
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("cache envelope wire field ") +
                              std::string(name) + " is not int64");
      return 0;
    }
  }

  int32_t i32_field(std::string_view name) {
    return static_cast<int32_t>(i64_field(name));
  }

  double double_field(std::string_view name) {
    const auto value = string_field(name);
    try {
      return std::stod(value);
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("cache envelope wire field ") +
                              std::string(name) + " is not double");
      return 0.0;
    }
  }

  bool bool_field(std::string_view name) {
    const auto value = string_field(name);
    if (value == "1") {
      return true;
    }
    if (value == "0") {
      return false;
    }
    m_diagnostics.push_back(std::string("cache envelope wire field ") +
                            std::string(name) + " is not bool");
    return false;
  }

  std::vector<std::string> string_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<std::string> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(string_field(name));
    }
    return values;
  }

  std::vector<int64_t> i64_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<int64_t> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(i64_field(name));
    }
    return values;
  }

  std::vector<int32_t> i32_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<int32_t> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(i32_field(name));
    }
    return values;
  }

  std::vector<uint32_t> u32_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<uint32_t> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(u32_field(name));
    }
    return values;
  }

  std::vector<size_t> size_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<size_t> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(size_field(name));
    }
    return values;
  }

private:
  std::string_view m_wire;
  size_t m_pos = 0;
  std::vector<std::string> m_diagnostics;
};

void append_runtime_param_contract(std::ostringstream &os,
                                   const RuntimeParamContract &contract) {
  append_field(os, std::to_string(contract.scalar_param_count));
  append_field(os, std::to_string(contract.shape_param_count));
  append_field(os, runtime_param_descriptor_payload_kind_to_string(
                       contract.descriptor_payload_kind));
  append_field(os, std::to_string(contract.params.size()));
  for (const auto &param : contract.params) {
    append_field(os, param.logical_name);
    append_field(os, runtime_param_kind_to_string(param.kind));
    append_field(os, param.abi_type);
    append_field(os, param.source_tensor);
  }
  append_vector(os, contract.runtime_param_names);
}

RuntimeParamContract read_runtime_param_contract(WireReader &reader) {
  RuntimeParamContract contract;
  contract.scalar_param_count = reader.size_field("runtime scalar count");
  contract.shape_param_count = reader.size_field("runtime shape count");
  contract.descriptor_payload_kind =
      runtime_param_descriptor_payload_kind_from_string(
          reader.string_field("runtime descriptor payload kind"));
  const auto count = reader.size_field("runtime param count");
  contract.params.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    RuntimeParamDescriptor param;
    param.logical_name = reader.string_field("runtime param logical name");
    param.kind =
        runtime_param_kind_from_string(reader.string_field("runtime param kind"));
    param.abi_type = reader.string_field("runtime param abi type");
    param.source_tensor = reader.string_field("runtime param source tensor");
    contract.params.push_back(std::move(param));
  }
  contract.runtime_param_names =
      reader.string_vector("runtime param names");
  return contract;
}

void append_tensor_contract(std::ostringstream &os,
                            const TensorContract &tensor) {
  append_field(os, tensor.logical_name);
  append_field(os, tensor.memory_region_id);
  append_field(os, tensor_contract_role_to_string(tensor.role));
  append_field(os, tensor.element_type);
  append_field(os, tensor.partial_shape);
  append_field(os, tensor.layout);
  append_field(os, tensor.storage_kind);
  append_field(os, tensor.lifetime_class);
  append_field(os, tensor.stateful_prebind_variable_id);
  append_field(os, tensor.stateful_prebind_shape_rule);
  append_field(os, std::to_string(tensor.stateful_prebind_shape_axis));
}

TensorContract read_tensor_contract(WireReader &reader) {
  TensorContract tensor;
  tensor.logical_name = reader.string_field("tensor logical name");
  tensor.memory_region_id = reader.string_field("tensor memory region id");
  tensor.role =
      tensor_contract_role_from_string(reader.string_field("tensor role"));
  tensor.element_type = reader.string_field("tensor element type");
  tensor.partial_shape = reader.string_field("tensor partial shape");
  tensor.layout = reader.string_field("tensor layout");
  tensor.storage_kind = reader.string_field("tensor storage kind");
  tensor.lifetime_class = reader.string_field("tensor lifetime class");
  tensor.stateful_prebind_variable_id =
      reader.string_field("tensor stateful variable id");
  tensor.stateful_prebind_shape_rule =
      reader.string_field("tensor stateful shape rule");
  tensor.stateful_prebind_shape_axis =
      reader.i64_field("tensor stateful shape axis");
  return tensor;
}

void append_memory_plan(std::ostringstream &os, const MemoryPlan &plan) {
  append_field(os, std::to_string(plan.schema_version));
  append_bool(os, plan.hidden_host_copies_allowed);
  append_field(os, std::to_string(plan.regions.size()));
  for (const auto &region : plan.regions) {
    append_field(os, region.region_id);
    append_field(os, region.logical_tensor_name);
    append_field(os, memory_region_kind_to_string(region.kind));
    append_field(os, region.element_type);
    append_field(os, region.partial_shape);
    append_field(os, region.layout);
    append_field(os, region.storage_kind);
    append_field(os, region.alias_group);
    append_field(os, std::to_string(region.lifetime.first_stage));
    append_field(os, std::to_string(region.lifetime.last_stage));
    append_bool(os, region.external_binding);
    append_bool(os, region.host_visible);
  }
  append_field(os, std::to_string(plan.alias_groups.size()));
  for (const auto &group : plan.alias_groups) {
    append_field(os, group.group_id);
    append_bool(os, group.output_aliasing);
    append_vector(os, group.region_ids);
  }
  append_field(os, std::to_string(plan.transient_arenas.size()));
  for (const auto &arena : plan.transient_arenas) {
    append_field(os, arena.arena_id);
    append_field(os, arena.storage_kind);
    append_vector(os, arena.region_ids);
  }
}

MemoryPlan read_memory_plan(WireReader &reader) {
  MemoryPlan plan;
  plan.schema_version = reader.u32_field("memory plan schema");
  plan.hidden_host_copies_allowed =
      reader.bool_field("memory plan hidden host copies");
  const auto region_count = reader.size_field("memory region count");
  plan.regions.reserve(region_count);
  for (size_t i = 0; i < region_count; ++i) {
    MemoryRegion region;
    region.region_id = reader.string_field("memory region id");
    region.logical_tensor_name =
        reader.string_field("memory region logical tensor name");
    region.kind =
        memory_region_kind_from_string(reader.string_field("memory region kind"));
    region.element_type = reader.string_field("memory region element type");
    region.partial_shape = reader.string_field("memory region partial shape");
    region.layout = reader.string_field("memory region layout");
    region.storage_kind = reader.string_field("memory region storage kind");
    region.alias_group = reader.string_field("memory region alias group");
    region.lifetime.first_stage =
        reader.size_field("memory region lifetime first");
    region.lifetime.last_stage =
        reader.size_field("memory region lifetime last");
    region.external_binding =
        reader.bool_field("memory region external binding");
    region.host_visible = reader.bool_field("memory region host visible");
    plan.regions.push_back(std::move(region));
  }
  const auto alias_count = reader.size_field("memory alias group count");
  plan.alias_groups.reserve(alias_count);
  for (size_t i = 0; i < alias_count; ++i) {
    AliasGroup group;
    group.group_id = reader.string_field("memory alias group id");
    group.output_aliasing =
        reader.bool_field("memory alias group output aliasing");
    group.region_ids = reader.string_vector("memory alias group regions");
    plan.alias_groups.push_back(std::move(group));
  }
  const auto arena_count = reader.size_field("memory transient arena count");
  plan.transient_arenas.reserve(arena_count);
  for (size_t i = 0; i < arena_count; ++i) {
    TransientArena arena;
    arena.arena_id = reader.string_field("memory transient arena id");
    arena.storage_kind =
        reader.string_field("memory transient arena storage kind");
    arena.region_ids = reader.string_vector("memory transient arena regions");
    plan.transient_arenas.push_back(std::move(arena));
  }
  return plan;
}

void append_stage_record(std::ostringstream &os, const StageRecord &stage) {
  append_field(os, std::to_string(stage.stage_id));
  append_field(os, std::to_string(stage.stable_record_key));
  append_field(os, stage.source_node_name);
  append_field(os, stage.normalized_op_family);
  append_field(os, lowering_route_kind_to_string(stage.execution_kind));
  append_field(os, stage.backend_domain);
  append_field(os, stage.kernel_unit_id);
  append_field(os, stage.kernel_unit_kind);
  append_bool(os, stage.requires_runtime_shape_args);
  append_field(os, std::to_string(stage.inputs.size()));
  for (const auto &input : stage.inputs) {
    append_tensor_contract(os, input);
  }
  append_field(os, std::to_string(stage.outputs.size()));
  for (const auto &output : stage.outputs) {
    append_tensor_contract(os, output);
  }
  append_runtime_param_contract(os, stage.runtime_params);
  append_field(os, stage.runtime_shape.rule);
  append_integral_vector(os, stage.runtime_shape.i64_metadata);
  append_field(os, stateful_effect_kind_to_string(stage.stateful_effect.kind));
  append_field(os, stage.stateful_effect.variable_id);
  append_field(os, lowering_route_kind_to_string(stage.dispatch.execution_kind));
  append_field(os, stage.dispatch.backend_domain);
  append_field(os, stage.dispatch.kernel_unit_id);
  append_field(os, stage.dispatch.kernel_unit_kind);
  append_field(os, stage.dispatch.dispatch_source);
  append_bool(os, stage.memory.hidden_host_copy_allowed);
  append_field(os, stage.memory.input_lifetime);
  append_field(os, stage.memory.output_lifetime);
  append_field(os, stage.memory.alias_group);
  append_field(os, std::to_string(stage.submission.stage_weight));
  append_field(os, std::to_string(stage.submission.macs_estimate));
  append_bool(os, stage.submission.dependency_extension_boundary);
  append_field(os, stage.handwritten_exception.ticket);
  append_field(os, stage.handwritten_exception.reason);
  append_field(os, stage.handwritten_exception.removal_condition);
  append_number(os, stage.profitability_score);
}

StageRecord read_stage_record(WireReader &reader) {
  StageRecord stage;
  stage.stage_id = reader.size_field("stage id");
  stage.stable_record_key = reader.u64_field("stage stable record key");
  stage.source_node_name = reader.string_field("stage source node name");
  stage.normalized_op_family =
      reader.string_field("stage normalized op family");
  stage.execution_kind =
      lowering_route_kind_from_string(reader.string_field("stage execution kind"));
  stage.backend_domain = reader.string_field("stage backend domain");
  stage.kernel_unit_id = reader.string_field("stage kernel unit id");
  stage.kernel_unit_kind = reader.string_field("stage kernel unit kind");
  stage.requires_runtime_shape_args =
      reader.bool_field("stage requires runtime shape args");
  const auto input_count = reader.size_field("stage input count");
  stage.inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    stage.inputs.push_back(read_tensor_contract(reader));
  }
  const auto output_count = reader.size_field("stage output count");
  stage.outputs.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    stage.outputs.push_back(read_tensor_contract(reader));
  }
  stage.runtime_params = read_runtime_param_contract(reader);
  stage.runtime_shape.rule = reader.string_field("stage runtime shape rule");
  stage.runtime_shape.i64_metadata =
      reader.i64_vector("stage runtime shape metadata");
  stage.stateful_effect.kind = stateful_effect_kind_from_string(
      reader.string_field("stage stateful effect kind"));
  stage.stateful_effect.variable_id =
      reader.string_field("stage stateful effect variable id");
  stage.dispatch.execution_kind = lowering_route_kind_from_string(
      reader.string_field("stage dispatch execution kind"));
  stage.dispatch.backend_domain =
      reader.string_field("stage dispatch backend domain");
  stage.dispatch.kernel_unit_id =
      reader.string_field("stage dispatch kernel id");
  stage.dispatch.kernel_unit_kind =
      reader.string_field("stage dispatch kernel kind");
  stage.dispatch.dispatch_source =
      reader.string_field("stage dispatch source");
  stage.memory.hidden_host_copy_allowed =
      reader.bool_field("stage memory hidden host copy");
  stage.memory.input_lifetime = reader.string_field("stage memory input lifetime");
  stage.memory.output_lifetime =
      reader.string_field("stage memory output lifetime");
  stage.memory.alias_group = reader.string_field("stage memory alias group");
  stage.submission.stage_weight =
      reader.u32_field("stage submission weight");
  stage.submission.macs_estimate =
      reader.u64_field("stage submission macs estimate");
  stage.submission.dependency_extension_boundary =
      reader.bool_field("stage submission dependency boundary");
  stage.handwritten_exception.ticket =
      reader.string_field("stage handwritten ticket");
  stage.handwritten_exception.reason =
      reader.string_field("stage handwritten reason");
  stage.handwritten_exception.removal_condition =
      reader.string_field("stage handwritten removal condition");
  stage.profitability_score = reader.double_field("stage profitability score");
  return stage;
}

void append_manifest(std::ostringstream &os, const ManifestBundle &manifest) {
  append_field(os, std::to_string(manifest.schema_version));
  append_field(os, manifest.target_fingerprint);
  append_memory_plan(os, manifest.memory_plan);
  append_field(os, std::to_string(manifest.stages.size()));
  for (const auto &stage : manifest.stages) {
    append_stage_record(os, stage);
  }
}

ManifestBundle read_manifest(WireReader &reader) {
  ManifestBundle manifest;
  manifest.schema_version = reader.u32_field("manifest schema");
  manifest.target_fingerprint = reader.string_field("manifest target");
  manifest.memory_plan = read_memory_plan(reader);
  const auto stage_count = reader.size_field("manifest stage count");
  manifest.stages.reserve(stage_count);
  for (size_t i = 0; i < stage_count; ++i) {
    manifest.stages.push_back(read_stage_record(reader));
  }
  return manifest;
}

void append_launch_plan(std::ostringstream &os,
                        const KernelLaunchPlanDescriptor &plan) {
  append_bool(os, plan.valid);
  append_vector(os, plan.buffer_roles);
  append_integral_vector(os, plan.direct_input_indices);
  append_integral_vector(os, plan.input_indices);
  append_field(os, std::to_string(plan.input_arg_count));
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
  plan.scalar_arg_kinds =
      reader.u32_vector("launch plan scalar arg kinds");
  return plan;
}

void append_kernel_descriptor(std::ostringstream &os,
                              const KernelDescriptor &kernel) {
  append_field(os, kernel.kernel_id);
  append_field(os, kernel.op_family);
  append_field(os, kernel.backend_domain);
  append_field(os, kernel_artifact_origin_to_string(kernel.origin));
  append_vector(os, kernel.tensor_roles);
  append_vector(os, kernel.scalar_roles);
  append_field(os, kernel.layout_contract);
  append_field(os, kernel.precision_contract);
  append_field(os, kernel.dispatch_contract);
  append_field(os, kernel.runtime_shape_rule);
  append_integral_vector(os, kernel.runtime_shape_i64_metadata);
  append_bool(os, kernel.requires_runtime_shape_args);
  append_field(os, kernel.exception_ticket);
  append_field(os, kernel.exception_reason);
  append_field(os, kernel.exception_removal_condition);
}

KernelDescriptor read_kernel_descriptor(WireReader &reader) {
  KernelDescriptor kernel;
  kernel.kernel_id = reader.string_field("kernel id");
  kernel.op_family = reader.string_field("kernel op family");
  kernel.backend_domain = reader.string_field("kernel backend domain");
  kernel.origin = kernel_artifact_origin_from_string(
      reader.string_field("kernel origin"));
  kernel.tensor_roles = reader.string_vector("kernel tensor roles");
  kernel.scalar_roles = reader.string_vector("kernel scalar roles");
  kernel.layout_contract = reader.string_field("kernel layout contract");
  kernel.precision_contract = reader.string_field("kernel precision contract");
  kernel.dispatch_contract = reader.string_field("kernel dispatch contract");
  kernel.runtime_shape_rule = reader.string_field("kernel runtime shape rule");
  kernel.runtime_shape_i64_metadata =
      reader.i64_vector("kernel runtime shape metadata");
  kernel.requires_runtime_shape_args =
      reader.bool_field("kernel requires runtime shape args");
  kernel.exception_ticket = reader.string_field("kernel exception ticket");
  kernel.exception_reason = reader.string_field("kernel exception reason");
  kernel.exception_removal_condition =
      reader.string_field("kernel exception removal condition");
  return kernel;
}

void append_artifact_descriptor(std::ostringstream &os,
                                const KernelArtifactDescriptor &descriptor) {
  append_field(os, std::to_string(descriptor.stage_record_key));
  append_field(os, descriptor.manifest_ref);
  append_field(os, descriptor.abi_fingerprint);
  append_field(os, descriptor.artifact_key);
  append_kernel_descriptor(os, descriptor.kernel);
  append_field(os, kernel_artifact_payload_kind_to_string(
                        descriptor.payload_kind));
  append_field(os, descriptor.entry_point);
  append_field(os, descriptor.compile_options_key);
  append_field(os, std::to_string(descriptor.abi_arg_count));
  append_field(os, std::to_string(descriptor.abi_output_arg_count));
  append_field(os, std::to_string(descriptor.runtime_param_buffer_count));
  append_field(os, runtime_param_descriptor_payload_kind_to_string(
                       descriptor.runtime_param_payload_kind));
  append_integral_vector(os, descriptor.runtime_param_i64_metadata);
  append_bool(os, descriptor.runtime_param_reduce_keep_dims);
  append_bool(os, descriptor.runtime_param_reduce_keep_dims_valid);
  append_launch_plan(os, descriptor.launch_plan);
  append_bool(os, descriptor.optional_cache_payload_allowed);
}

KernelArtifactDescriptor read_artifact_descriptor(WireReader &reader) {
  KernelArtifactDescriptor descriptor;
  descriptor.stage_record_key = reader.u64_field("artifact stage key");
  descriptor.manifest_ref = reader.string_field("artifact manifest ref");
  descriptor.abi_fingerprint = reader.string_field("artifact abi fingerprint");
  descriptor.artifact_key = reader.string_field("artifact key");
  descriptor.kernel = read_kernel_descriptor(reader);
  descriptor.payload_kind = kernel_artifact_payload_kind_from_string(
      reader.string_field("artifact payload kind"));
  descriptor.entry_point = reader.string_field("artifact entry point");
  descriptor.compile_options_key =
      reader.string_field("artifact compile options key");
  descriptor.abi_arg_count = reader.u32_field("artifact abi arg count");
  descriptor.abi_output_arg_count =
      reader.u32_field("artifact abi output arg count");
  descriptor.runtime_param_buffer_count =
      reader.u32_field("artifact runtime param buffer count");
  descriptor.runtime_param_payload_kind =
      runtime_param_descriptor_payload_kind_from_string(
          reader.string_field("artifact runtime param payload kind"));
  descriptor.runtime_param_i64_metadata =
      reader.i64_vector("artifact runtime param metadata");
  descriptor.runtime_param_reduce_keep_dims =
      reader.bool_field("artifact reduce keep dims");
  descriptor.runtime_param_reduce_keep_dims_valid =
      reader.bool_field("artifact reduce keep dims valid");
  descriptor.launch_plan = read_launch_plan(reader);
  descriptor.optional_cache_payload_allowed =
      reader.bool_field("artifact optional cache payload allowed");
  return descriptor;
}

void append_cache_key(std::ostringstream &os, const CacheKey &key) {
  append_field(os, key.model_fingerprint);
  append_field(os, key.manifest_hash);
  append_field(os, key.target_fingerprint);
  append_field(os, key.backend_capabilities_fingerprint);
  append_field(os, key.compiler_revision);
  append_field(os, key.backend_compiler_revision);
  append_field(os, key.driver_identity);
  append_field(os, key.compile_options_hash);
  append_vector(os, key.kernel_unit_versions);
  append_field(os, key.stable_key);
}

CacheKey read_cache_key(WireReader &reader) {
  CacheKey key;
  key.model_fingerprint = reader.string_field("cache key model fingerprint");
  key.manifest_hash = reader.string_field("cache key manifest hash");
  key.target_fingerprint = reader.string_field("cache key target fingerprint");
  key.backend_capabilities_fingerprint =
      reader.string_field("cache key capabilities fingerprint");
  key.compiler_revision = reader.string_field("cache key compiler revision");
  key.backend_compiler_revision =
      reader.string_field("cache key backend compiler revision");
  key.driver_identity = reader.string_field("cache key driver identity");
  key.compile_options_hash =
      reader.string_field("cache key compile options hash");
  key.kernel_unit_versions =
      reader.string_vector("cache key kernel unit versions");
  key.stable_key = reader.string_field("cache key stable key");
  return key;
}

void append_backend_payload(std::ostringstream &os,
                            const CacheBackendPayloadRecord &payload) {
  append_field(os, payload.artifact_key);
  append_field(os, payload.backend_domain);
  append_field(os, payload.payload_kind);
  append_field(os, payload.source_id);
  append_field(os, payload.entry_point);
  append_field(os, payload.payload_identity);
  append_field(os, payload.source_language);
  append_field(os, payload.source);
  append_field(os, payload.payload_format);
  append_field(os, payload.payload_data);
  append_bool(os, payload.optional);
}

CacheBackendPayloadRecord read_backend_payload(WireReader &reader) {
  CacheBackendPayloadRecord payload;
  payload.artifact_key = reader.string_field("payload artifact key");
  payload.backend_domain = reader.string_field("payload backend domain");
  payload.payload_kind = reader.string_field("payload kind");
  payload.source_id = reader.string_field("payload source id");
  payload.entry_point = reader.string_field("payload entry point");
  payload.payload_identity = reader.string_field("payload identity");
  payload.source_language = reader.string_field("payload source language");
  payload.source = reader.string_field("payload source");
  payload.payload_format = reader.string_field("payload format");
  payload.payload_data = reader.string_field("payload data");
  payload.optional = reader.bool_field("payload optional");
  return payload;
}

} // namespace

std::string serialize_cache_envelope(const CacheEnvelope &envelope) {
  std::ostringstream os;
  append_field(os, "GFX_CACHE_ENVELOPE");
  append_field(os, "5");
  append_field(os, std::to_string(envelope.schema_version));
  append_cache_key(os, envelope.key);
  append_manifest(os, envelope.manifest);
  append_field(os, std::to_string(envelope.artifact_descriptors.size()));
  for (const auto &descriptor : envelope.artifact_descriptors) {
    append_artifact_descriptor(os, descriptor);
  }
  append_field(os, serialize_cache_materialization_contract(
                       envelope.materialization));
  append_field(os, std::to_string(envelope.backend_payloads.size()));
  for (const auto &payload : envelope.backend_payloads) {
    append_backend_payload(os, payload);
  }
  return os.str();
}

CacheEnvelopeWireResult deserialize_cache_envelope(std::string_view wire) {
  CacheEnvelopeWireResult result;
  WireReader reader(wire);
  const auto magic = reader.string_field("cache envelope magic");
  const auto wire_version = reader.string_field("cache envelope wire version");
  if (magic != "GFX_CACHE_ENVELOPE") {
    result.diagnostics.emplace_back("cache envelope wire magic mismatch");
  }
  if (wire_version != "5") {
    result.diagnostics.emplace_back("cache envelope wire version mismatch");
  }
  result.envelope.schema_version =
      reader.u32_field("cache envelope schema version");
  result.envelope.key = read_cache_key(reader);
  result.envelope.manifest = read_manifest(reader);
  const auto artifact_count =
      reader.size_field("cache envelope artifact count");
  result.envelope.artifact_descriptors.reserve(artifact_count);
  for (size_t i = 0; i < artifact_count; ++i) {
    result.envelope.artifact_descriptors.push_back(
        read_artifact_descriptor(reader));
  }
  result.envelope.materialization =
      deserialize_cache_materialization_contract(
          reader.string_field("cache envelope materialization contract"),
          result.diagnostics);
  const auto payload_count = reader.size_field("cache envelope payload count");
  result.envelope.backend_payloads.reserve(payload_count);
  for (size_t i = 0; i < payload_count; ++i) {
    result.envelope.backend_payloads.push_back(read_backend_payload(reader));
  }
  auto read_diagnostics = reader.take_diagnostics();
  result.diagnostics.insert(result.diagnostics.end(),
                            read_diagnostics.begin(), read_diagnostics.end());
  return result;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
