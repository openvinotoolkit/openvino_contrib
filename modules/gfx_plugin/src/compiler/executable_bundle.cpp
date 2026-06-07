// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/executable_bundle.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>

#include "common/constant_tensor_evaluator.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

namespace {

constexpr uint32_t kExecutableBundleSchemaVersion = 1;

uint64_t stable_hash64(std::string_view value) noexcept {
  uint64_t hash = 14695981039346656037ull;
  for (const unsigned char c : value) {
    hash ^= c;
    hash *= 1099511628211ull;
  }
  return hash;
}

void append_field(std::ostringstream &os, std::string_view value) {
  os << value.size() << ":" << value << ";";
}

void append_bool(std::ostringstream &os, bool value) {
  append_field(os, value ? "1" : "0");
}

void append_u32(std::ostringstream &os, uint32_t value) {
  append_field(os, std::to_string(value));
}

void append_size(std::ostringstream &os, size_t value) {
  append_field(os, std::to_string(value));
}

void append_i32_vector(std::ostringstream &os,
                       const std::vector<int32_t> &values) {
  append_field(os, std::to_string(values.size()));
  for (const auto value : values) {
    append_field(os, std::to_string(value));
  }
}

void append_i64_vector(std::ostringstream &os,
                       const std::vector<int64_t> &values) {
  append_field(os, std::to_string(values.size()));
  for (const auto value : values) {
    append_field(os, std::to_string(value));
  }
}

void append_u32_vector(std::ostringstream &os,
                       const std::vector<uint32_t> &values) {
  append_field(os, std::to_string(values.size()));
  for (const auto value : values) {
    append_field(os, std::to_string(value));
  }
}

void append_size_vector(std::ostringstream &os,
                        const std::vector<size_t> &values) {
  append_field(os, std::to_string(values.size()));
  for (const auto value : values) {
    append_field(os, std::to_string(value));
  }
}

void append_string_vector(std::ostringstream &os,
                          const std::vector<std::string> &values) {
  append_field(os, std::to_string(values.size()));
  for (const auto &value : values) {
    append_field(os, value);
  }
}

void append_launch_plan(std::ostringstream &os,
                        const KernelLaunchPlanDescriptor &plan) {
  append_bool(os, plan.valid);
  append_string_vector(os, plan.buffer_roles);
  append_size_vector(os, plan.direct_input_indices);
  append_size_vector(os, plan.input_indices);
  append_size(os, plan.input_arg_count);
  append_i32_vector(os, plan.operand_kinds);
  append_i32_vector(os, plan.operand_arg_indices);
  append_i32_vector(os, plan.scalar_args);
  append_u32_vector(os, plan.scalar_arg_kinds);
}

std::string hex64(uint64_t value) {
  std::ostringstream os;
  os << std::hex << std::setw(16) << std::setfill('0') << value;
  return os.str();
}

bool same_string(std::string_view lhs, const std::string &rhs) {
  return lhs.size() == rhs.size() && lhs.compare(rhs) == 0;
}

KernelArtifactOrigin origin_from_route(LoweringRouteKind kind) noexcept {
  switch (kind) {
  case LoweringRouteKind::Common:
    return KernelArtifactOrigin::Common;
  case LoweringRouteKind::Metadata:
    return KernelArtifactOrigin::Metadata;
  case LoweringRouteKind::VendorPrimitive:
    return KernelArtifactOrigin::VendorPrimitive;
  case LoweringRouteKind::GeneratedKernel:
    return KernelArtifactOrigin::Generated;
  case LoweringRouteKind::HandwrittenKernelException:
    return KernelArtifactOrigin::HandwrittenException;
  case LoweringRouteKind::Unsupported:
  default:
    return KernelArtifactOrigin::Unknown;
  }
}

KernelArtifactPayloadKind payload_kind_for_stage(const StageRecord &stage) {
  switch (stage.execution_kind) {
  case LoweringRouteKind::Common:
  case LoweringRouteKind::Metadata:
    return KernelArtifactPayloadKind::None;
  case LoweringRouteKind::VendorPrimitive:
    return KernelArtifactPayloadKind::VendorDescriptor;
  case LoweringRouteKind::GeneratedKernel:
  case LoweringRouteKind::HandwrittenKernelException:
    if (stage.backend_domain == "opencl") {
      return KernelArtifactPayloadKind::OpenClSource;
    }
    if (stage.backend_domain == "metal" ||
        stage.backend_domain == "apple_msl") {
      return KernelArtifactPayloadKind::MslSource;
    }
    return KernelArtifactPayloadKind::None;
  case LoweringRouteKind::Unsupported:
  default:
    return KernelArtifactPayloadKind::None;
  }
}

bool payload_kind_requires_materialized_payload(
    KernelArtifactPayloadKind kind) noexcept {
  return kind == KernelArtifactPayloadKind::VendorDescriptor ||
         kind == KernelArtifactPayloadKind::MslSource ||
         kind == KernelArtifactPayloadKind::OpenClSource;
}

bool source_payload_kind(KernelArtifactPayloadKind kind) noexcept {
  return kind == KernelArtifactPayloadKind::MslSource ||
         kind == KernelArtifactPayloadKind::OpenClSource;
}

size_t count_launch_plan_role(const KernelLaunchPlanDescriptor &plan,
                              std::string_view role_name) {
  return static_cast<size_t>(std::count(plan.buffer_roles.begin(),
                                        plan.buffer_roles.end(),
                                        std::string(role_name)));
}

std::string make_kernel_abi_fingerprint(const KernelDescriptor &kernel) {
  std::ostringstream material;
  append_field(material, kernel.kernel_id);
  append_field(material, kernel.op_family);
  append_field(material, kernel.backend_domain);
  append_field(material, kernel_artifact_origin_to_string(kernel.origin));
  append_field(material, kernel.layout_contract);
  append_field(material, kernel.precision_contract);
  append_field(material, kernel.dispatch_contract);
  append_field(material, kernel.runtime_shape_rule);
  append_i64_vector(material, kernel.runtime_shape_i64_metadata);
  append_bool(material, kernel.requires_runtime_shape_args);
  for (const auto &role : kernel.tensor_roles) {
    append_field(material, role);
  }
  for (const auto &role : kernel.scalar_roles) {
    append_field(material, role);
  }
  append_field(material, kernel.exception_ticket);
  append_field(material, kernel.exception_reason);
  append_field(material, kernel.exception_removal_condition);
  return hex64(stable_hash64(material.str()));
}

std::string
make_kernel_abi_fingerprint(const KernelArtifactDescriptor &descriptor) {
  std::ostringstream material;
  append_field(material, make_kernel_abi_fingerprint(descriptor.kernel));
  append_u32(material, descriptor.runtime_param_buffer_count);
  append_i64_vector(material, descriptor.runtime_param_i64_metadata);
  append_bool(material, descriptor.runtime_param_reduce_keep_dims);
  append_bool(material, descriptor.runtime_param_reduce_keep_dims_valid);
  append_launch_plan(material, descriptor.launch_plan);
  return hex64(stable_hash64(material.str()));
}

std::string make_manifest_ref(uint64_t stage_record_key,
                              std::string_view kernel_unit_id,
                              std::string_view abi_fingerprint) {
  std::ostringstream os;
  os << stage_record_key << ":" << kernel_unit_id << ":" << abi_fingerprint;
  return os.str();
}

std::string make_manifest_ref(const StageRecord &stage,
                              std::string_view abi_fingerprint) {
  return make_manifest_ref(stage.stable_record_key, stage.kernel_unit_id,
                           abi_fingerprint);
}

std::string make_artifact_key(uint64_t stage_record_key,
                              std::string_view backend_domain,
                              std::string_view kernel_unit_id,
                              KernelArtifactPayloadKind payload_kind,
                              std::string_view abi_fingerprint) {
  std::ostringstream material;
  append_field(material, std::to_string(stage_record_key));
  append_field(material, backend_domain);
  append_field(material, kernel_unit_id);
  append_field(material, kernel_artifact_payload_kind_to_string(payload_kind));
  append_field(material, abi_fingerprint);
  return hex64(stable_hash64(material.str()));
}

std::string make_artifact_key(const StageRecord &stage,
                              KernelArtifactPayloadKind payload_kind,
                              std::string_view abi_fingerprint) {
  return make_artifact_key(stage.stable_record_key, stage.backend_domain,
                           stage.kernel_unit_id, payload_kind, abi_fingerprint);
}

KernelArtifactDescriptor make_artifact_descriptor(const StageRecord &stage) {
  KernelArtifactDescriptor descriptor;
  descriptor.stage_record_key = stage.stable_record_key;
  descriptor.payload_kind = payload_kind_for_stage(stage);
  descriptor.entry_point = stage.kernel_unit_id;
  descriptor.compile_options_key = stage.dispatch.dispatch_source;

  descriptor.kernel.kernel_id = stage.kernel_unit_id;
  descriptor.kernel.op_family = stage.normalized_op_family;
  descriptor.kernel.backend_domain = stage.backend_domain;
  descriptor.kernel.origin = origin_from_route(stage.execution_kind);
  descriptor.kernel.dispatch_contract = stage.dispatch.dispatch_source;
  descriptor.kernel.runtime_shape_rule = stage.runtime_shape.rule;
  descriptor.kernel.runtime_shape_i64_metadata =
      stage.runtime_shape.i64_metadata;
  descriptor.kernel.requires_runtime_shape_args =
      stage.requires_runtime_shape_args;
  if (!stage.inputs.empty()) {
    descriptor.kernel.layout_contract = stage.inputs.front().layout;
    descriptor.kernel.precision_contract = stage.inputs.front().element_type;
  } else if (!stage.outputs.empty()) {
    descriptor.kernel.layout_contract = stage.outputs.front().layout;
    descriptor.kernel.precision_contract = stage.outputs.front().element_type;
  }
  descriptor.kernel.tensor_roles.reserve(stage.inputs.size() +
                                         stage.outputs.size());
  for (const auto &input : stage.inputs) {
    descriptor.kernel.tensor_roles.push_back(
        std::string(tensor_contract_role_to_string(input.role)));
  }
  for (const auto &output : stage.outputs) {
    descriptor.kernel.tensor_roles.push_back(
        std::string(tensor_contract_role_to_string(output.role)));
  }
  descriptor.kernel.scalar_roles.reserve(stage.runtime_params.params.size());
  for (const auto &param : stage.runtime_params.params) {
    descriptor.kernel.scalar_roles.push_back(
        std::string(runtime_param_kind_to_string(param.kind)) + ":" +
        param.abi_type);
  }
  descriptor.abi_arg_count =
      static_cast<uint32_t>(descriptor.kernel.tensor_roles.size() +
                            descriptor.kernel.scalar_roles.size());
  descriptor.abi_output_arg_count = static_cast<uint32_t>(stage.outputs.size());
  if (descriptor.kernel.origin == KernelArtifactOrigin::HandwrittenException) {
    descriptor.kernel.exception_ticket = stage.handwritten_exception.ticket;
    descriptor.kernel.exception_reason = stage.handwritten_exception.reason;
    descriptor.kernel.exception_removal_condition =
        stage.handwritten_exception.removal_condition;
  }
  descriptor.abi_fingerprint = make_kernel_abi_fingerprint(descriptor);
  descriptor.manifest_ref =
      make_manifest_ref(stage, descriptor.abi_fingerprint);
  descriptor.artifact_key = make_artifact_key(stage, descriptor.payload_kind,
                                              descriptor.abi_fingerprint);
  return descriptor;
}

std::shared_ptr<const KernelArtifactPayload>
materialize_payload_for_stage(const KernelArtifactDescriptor &descriptor,
                              const PlannedOperation &op,
                              const KernelArtifactPayloadResolver &resolver) {
  return resolver ? resolver(descriptor, op) : nullptr;
}

bool finalize_descriptor_for_stage(
    KernelArtifactDescriptor &descriptor, const PlannedOperation &op,
    const KernelArtifactDescriptorResolver &resolver) {
  if (!resolver) {
    finalize_kernel_artifact_descriptor_identity(descriptor);
    return true;
  }
  return resolver(descriptor, op);
}

std::vector<KernelArtifactConstTensor>
materialize_const_tensors_for_stage(const PlannedOperation &op) {
  std::vector<KernelArtifactConstTensor> tensors;
  if (!op.source_node) {
    return tensors;
  }

  for (size_t input_idx = 0; input_idx < op.source_node->get_input_size();
       ++input_idx) {
    auto const_tensor = gfx_evaluate_constant_source_tensor(
        op.source_node->input_value(input_idx));
    if (!const_tensor.has_value()) {
      continue;
    }

    KernelArtifactConstTensor tensor;
    tensor.source_input_index = input_idx;
    tensor.logical_name =
        op.node_name + ".const_input" + std::to_string(input_idx);
    tensor.element_type = const_tensor->get_element_type().get_type_name();
    tensor.shape.assign(const_tensor->get_shape().begin(),
                        const_tensor->get_shape().end());
    const auto byte_size = const_tensor->get_byte_size();
    tensor.bytes.resize(byte_size);
    if (byte_size != 0) {
      std::memcpy(tensor.bytes.data(), const_tensor->data(), byte_size);
    }
    tensors.push_back(std::move(tensor));
  }
  return tensors;
}

} // namespace

std::string_view
kernel_artifact_origin_to_string(KernelArtifactOrigin origin) noexcept {
  switch (origin) {
  case KernelArtifactOrigin::Common:
    return "common";
  case KernelArtifactOrigin::Metadata:
    return "metadata";
  case KernelArtifactOrigin::VendorPrimitive:
    return "vendor_primitive";
  case KernelArtifactOrigin::Generated:
    return "generated";
  case KernelArtifactOrigin::HandwrittenException:
    return "handwritten_exception";
  case KernelArtifactOrigin::Unknown:
  default:
    return "unknown";
  }
}

std::string_view kernel_artifact_payload_kind_to_string(
    KernelArtifactPayloadKind kind) noexcept {
  switch (kind) {
  case KernelArtifactPayloadKind::None:
    return "none";
  case KernelArtifactPayloadKind::VendorDescriptor:
    return "vendor_descriptor";
  case KernelArtifactPayloadKind::MslSource:
    return "msl_source";
  case KernelArtifactPayloadKind::OpenClSource:
    return "opencl_source";
  }
  return "none";
}

void finalize_kernel_artifact_descriptor_identity(
    KernelArtifactDescriptor &descriptor) {
  descriptor.abi_fingerprint = make_kernel_abi_fingerprint(descriptor);
  descriptor.manifest_ref = make_manifest_ref(descriptor.stage_record_key,
                                              descriptor.kernel.kernel_id,
                                              descriptor.abi_fingerprint);
  descriptor.artifact_key = make_artifact_key(
      descriptor.stage_record_key, descriptor.kernel.backend_domain,
      descriptor.kernel.kernel_id, descriptor.payload_kind,
      descriptor.abi_fingerprint);
}

ExecutableBundleVerificationResult ExecutableBundle::verify() const {
  ExecutableBundleVerificationResult result;
  if (schema_version != kExecutableBundleSchemaVersion ||
      target_fingerprint.empty()) {
    result.diagnostics.emplace_back("executable bundle header is incomplete");
  }
  const auto manifest_result = manifest.verify();
  for (const auto &diagnostic : manifest_result.diagnostics) {
    result.diagnostics.push_back("manifest: " + diagnostic);
  }
  const auto memory_result = memory_plan.verify();
  for (const auto &diagnostic : memory_result.diagnostics) {
    result.diagnostics.push_back("memory plan: " + diagnostic);
  }
  if (target_fingerprint != manifest.target_fingerprint) {
    result.diagnostics.emplace_back("executable bundle target drift");
  }
  if (make_memory_plan_fingerprint(memory_plan) !=
      make_memory_plan_fingerprint(manifest.memory_plan)) {
    result.diagnostics.emplace_back("executable bundle memory plan drift");
  }
  if (stages.size() != manifest.stages.size()) {
    result.diagnostics.emplace_back("executable bundle stage count drift");
  }
  if (artifact_descriptors.size() != manifest.stages.size()) {
    result.diagnostics.emplace_back("executable bundle artifact count drift");
  }
  const size_t count = std::min(stages.size(), manifest.stages.size());
  for (size_t i = 0; i < count; ++i) {
    const auto &executable_stage = stages[i];
    const auto &manifest_stage = manifest.stages[i];
    if (executable_stage.stage_record_key != manifest_stage.stable_record_key ||
        executable_stage.kernel_unit_id != manifest_stage.kernel_unit_id ||
        executable_stage.kernel_unit_kind != manifest_stage.kernel_unit_kind ||
        executable_stage.execution_kind != manifest_stage.execution_kind) {
      result.diagnostics.push_back("executable bundle stage drift at " +
                                   std::to_string(i));
    }
    if (executable_stage.artifact_descriptor_index >=
        artifact_descriptors.size()) {
      result.diagnostics.push_back(
          "executable bundle artifact index out of range at " +
          std::to_string(i));
      continue;
    }
    const auto &artifact =
        artifact_descriptors[executable_stage.artifact_descriptor_index];
    if (artifact.stage_record_key != manifest_stage.stable_record_key ||
        artifact.kernel.kernel_id != manifest_stage.kernel_unit_id ||
        artifact.kernel.backend_domain != manifest_stage.backend_domain ||
        artifact.kernel.runtime_shape_rule !=
            manifest_stage.runtime_shape.rule ||
        artifact.kernel.runtime_shape_i64_metadata !=
            manifest_stage.runtime_shape.i64_metadata ||
        artifact.kernel.origin !=
            origin_from_route(manifest_stage.execution_kind)) {
      result.diagnostics.push_back("executable bundle artifact drift at " +
                                   std::to_string(i));
    }
    if (artifact.manifest_ref.empty()) {
      result.diagnostics.push_back(
          "executable bundle artifact manifest ref is empty at " +
          std::to_string(i));
    }
    if (artifact.abi_fingerprint.empty()) {
      result.diagnostics.push_back(
          "executable bundle artifact ABI fingerprint is empty at " +
          std::to_string(i));
    }
    if (artifact.artifact_key.empty()) {
      result.diagnostics.push_back(
          "executable bundle artifact key is empty at " + std::to_string(i));
    }
    if (artifact.kernel.origin == KernelArtifactOrigin::HandwrittenException &&
        (artifact.kernel.exception_ticket.empty() ||
         artifact.kernel.exception_reason.empty() ||
         artifact.kernel.exception_removal_condition.empty())) {
      result.diagnostics.push_back(
          "handwritten artifact is missing exception contract at " +
          std::to_string(i));
    }
    if (artifact.kernel.origin != KernelArtifactOrigin::HandwrittenException &&
        (!artifact.kernel.exception_ticket.empty() ||
         !artifact.kernel.exception_reason.empty() ||
         !artifact.kernel.exception_removal_condition.empty())) {
      result.diagnostics.push_back(
          "non-exception artifact carries exception contract at " +
          std::to_string(i));
    }
    if ((artifact.payload_kind == KernelArtifactPayloadKind::MslSource ||
         artifact.payload_kind == KernelArtifactPayloadKind::OpenClSource) &&
        (artifact.abi_arg_count == 0 || artifact.abi_output_arg_count == 0)) {
      result.diagnostics.push_back("source artifact ABI counts are empty at " +
                                   std::to_string(i));
    }
    if (source_payload_kind(artifact.payload_kind)) {
      if (!artifact.launch_plan.valid ||
          artifact.launch_plan.buffer_roles.empty()) {
        result.diagnostics.push_back(
            "source artifact launch plan is incomplete at " +
            std::to_string(i));
      }
      if (artifact.launch_plan.buffer_roles.size() != artifact.abi_arg_count) {
        result.diagnostics.push_back(
            "source artifact launch-plan ABI count drift at " +
            std::to_string(i));
      }
      if (count_launch_plan_role(artifact.launch_plan, "tensor_output") !=
          artifact.abi_output_arg_count) {
        result.diagnostics.push_back(
            "source artifact launch-plan output count drift at " +
            std::to_string(i));
      }
    }
    if (artifact.payload_kind == KernelArtifactPayloadKind::VendorDescriptor &&
        artifact.kernel.origin != KernelArtifactOrigin::VendorPrimitive) {
      result.diagnostics.push_back("vendor payload has non-vendor origin at " +
                                   std::to_string(i));
    }
    if (payload_kind_requires_materialized_payload(artifact.payload_kind) &&
        !find_artifact_payload(artifact.artifact_key)) {
      result.diagnostics.push_back(
          "executable bundle artifact requires a materialized payload at " +
          std::to_string(i));
    }
  }
  for (size_t i = 0; i < artifact_payloads.size(); ++i) {
    const auto &payload_record = artifact_payloads[i];
    if (payload_record.artifact_descriptor_index >=
        artifact_descriptors.size()) {
      result.diagnostics.push_back(
          "executable bundle payload index out of range at " +
          std::to_string(i));
      continue;
    }
    const auto &descriptor =
        artifact_descriptors[payload_record.artifact_descriptor_index];
    if (payload_record.artifact_key != descriptor.artifact_key) {
      result.diagnostics.push_back(
          "executable bundle payload artifact key drift at " +
          std::to_string(i));
    }
    if (!payload_record.payload || !payload_record.payload->valid()) {
      result.diagnostics.push_back(
          "executable bundle payload is missing or invalid at " +
          std::to_string(i));
      continue;
    }
    if (payload_record.payload->payload_kind() != descriptor.payload_kind ||
        !same_string(payload_record.payload->backend_domain(),
                     descriptor.kernel.backend_domain) ||
        !same_string(payload_record.payload->source_id(),
                     descriptor.kernel.kernel_id) ||
        !same_string(payload_record.payload->entry_point(),
                     descriptor.entry_point)) {
      result.diagnostics.push_back(
          "executable bundle payload identity drift at " + std::to_string(i));
    }
  }
  return result;
}

bool ExecutableBundle::valid() const { return verify().valid(); }

std::shared_ptr<const KernelArtifactPayload>
ExecutableBundle::find_artifact_payload(const std::string &artifact_key) const {
  for (const auto &record : artifact_payloads) {
    if (record.artifact_key == artifact_key) {
      return record.payload;
    }
  }
  return {};
}

const std::vector<KernelArtifactConstTensor> *
ExecutableBundle::find_artifact_const_tensors(
    const std::string &artifact_key) const {
  for (const auto &record : artifact_payloads) {
    if (record.artifact_key == artifact_key) {
      return &record.const_tensors;
    }
  }
  return nullptr;
}

ExecutableBundleBuilder::ExecutableBundleBuilder(
    KernelArtifactPayloadResolver resolver)
    : m_payload_resolver(std::move(resolver)) {}

ExecutableBundleBuilder::ExecutableBundleBuilder(
    KernelArtifactDescriptorResolver descriptor_resolver,
    KernelArtifactPayloadResolver payload_resolver)
    : m_descriptor_resolver(std::move(descriptor_resolver)),
      m_payload_resolver(std::move(payload_resolver)) {}

ExecutableBundle
ExecutableBundleBuilder::build(const ManifestBundle &manifest) const {
  ExecutableBundle bundle;
  bundle.schema_version = kExecutableBundleSchemaVersion;
  bundle.target_fingerprint = manifest.target_fingerprint;
  bundle.manifest = manifest;
  bundle.memory_plan = manifest.memory_plan;
  bundle.stages.reserve(manifest.stages.size());
  for (const auto &stage : manifest.stages) {
    ExecutableStageRecord record;
    record.stage_record_key = stage.stable_record_key;
    record.kernel_unit_id = stage.kernel_unit_id;
    record.kernel_unit_kind = stage.kernel_unit_kind;
    record.execution_kind = stage.execution_kind;
    record.artifact_descriptor_index = bundle.artifact_descriptors.size();
    bundle.artifact_descriptors.push_back(make_artifact_descriptor(stage));
    bundle.stages.push_back(std::move(record));
  }
  return bundle;
}

ExecutableBundle
ExecutableBundleBuilder::build(const ManifestBundle &manifest,
                               const LoweringPlan &lowering_plan) const {
  auto bundle = build(manifest);
  const size_t count =
      std::min(bundle.stages.size(), lowering_plan.operations.size());
  for (size_t i = 0; i < count; ++i) {
    const auto descriptor_index = bundle.stages[i].artifact_descriptor_index;
    if (descriptor_index >= bundle.artifact_descriptors.size()) {
      continue;
    }
    auto &descriptor = bundle.artifact_descriptors[descriptor_index];
    if (!finalize_descriptor_for_stage(descriptor, lowering_plan.operations[i],
                                       m_descriptor_resolver)) {
      continue;
    }
    auto payload = materialize_payload_for_stage(
        descriptor, lowering_plan.operations[i], m_payload_resolver);
    if (!payload) {
      continue;
    }
    KernelArtifactPayloadRecord record;
    record.artifact_descriptor_index = descriptor_index;
    record.artifact_key = descriptor.artifact_key;
    record.payload = std::move(payload);
    record.const_tensors =
        materialize_const_tensors_for_stage(lowering_plan.operations[i]);
    bundle.artifact_payloads.push_back(std::move(record));
  }
  return bundle;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
