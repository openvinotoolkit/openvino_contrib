// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/manifest.hpp"

#include <sstream>
#include <utility>

namespace ov {
namespace gfx_plugin {
namespace compiler {

namespace {

constexpr uint32_t kManifestSchemaVersion = 2;

uint64_t stable_hash64(std::string_view value) noexcept {
  uint64_t hash = 14695981039346656037ull;
  for (const unsigned char c : value) {
    hash ^= c;
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string stage_key_material(const LoweringPlan &plan,
                               const PlannedOperation &op, size_t stage_id) {
  std::ostringstream os;
  os << plan.target.fingerprint() << "#" << stage_id << "#" << op.node_name
     << "#" << op.type_name << "#" << op.kernel_unit.manifest_key();
  return os.str();
}

std::string backend_domain(const BackendTarget &target) {
  return target.backend_id();
}

bool shape_is_dynamic(const std::string &partial_shape) {
  return partial_shape.find('?') != std::string::npos ||
         partial_shape.find("-1") != std::string::npos;
}

TensorContract make_tensor_contract(const PlannedOperation &op,
                                    TensorContractRole role, size_t index,
                                    std::string element_type,
                                    std::string partial_shape) {
  TensorContract contract;
  contract.logical_name =
      op.node_name +
      (role == TensorContractRole::TensorInput ? ".input" : ".output") +
      std::to_string(index);
  contract.role = role;
  contract.element_type = std::move(element_type);
  contract.partial_shape = std::move(partial_shape);
  contract.layout = std::string(tensor_layout_kind_to_string(op.layout.kind));
  contract.lifetime_class = role == TensorContractRole::TensorInput
                                ? "producer_or_external"
                                : "stage_output";
  return contract;
}

std::vector<TensorContract> make_input_contracts(const PlannedOperation &op) {
  std::vector<TensorContract> contracts;
  contracts.reserve(op.input_element_types.size());
  for (size_t i = 0; i < op.input_element_types.size(); ++i) {
    const std::string shape =
        i < op.input_shapes.size() ? op.input_shapes[i] : std::string{};
    auto contract = make_tensor_contract(op, TensorContractRole::TensorInput, i,
                                         op.input_element_types[i], shape);
    contracts.push_back(std::move(contract));
  }
  return contracts;
}

std::vector<TensorContract> make_output_contracts(const PlannedOperation &op) {
  std::vector<TensorContract> contracts;
  contracts.reserve(op.output_element_types.size());
  for (size_t i = 0; i < op.output_element_types.size(); ++i) {
    const std::string shape =
        i < op.output_shapes.size() ? op.output_shapes[i] : std::string{};
    contracts.push_back(
        make_tensor_contract(op, TensorContractRole::TensorOutput, i,
                             op.output_element_types[i], shape));
  }
  return contracts;
}

RuntimeParamContract make_runtime_param_contract(const StageRecord &stage) {
  RuntimeParamContract contract;
  for (const auto &tensor : stage.inputs) {
    if (shape_is_dynamic(tensor.partial_shape)) {
      RuntimeParamDescriptor param;
      param.logical_name = tensor.logical_name + ".shape";
      param.kind = RuntimeParamKind::Shape;
      param.abi_type = "shape_i64";
      param.source_tensor = tensor.logical_name;
      contract.params.push_back(std::move(param));
    }
  }
  for (const auto &tensor : stage.outputs) {
    if (shape_is_dynamic(tensor.partial_shape)) {
      RuntimeParamDescriptor param;
      param.logical_name = tensor.logical_name + ".shape";
      param.kind = RuntimeParamKind::Shape;
      param.abi_type = "shape_i64";
      param.source_tensor = tensor.logical_name;
      contract.params.push_back(std::move(param));
    }
  }
  for (const auto &param : contract.params) {
    contract.runtime_param_names.push_back(param.logical_name);
    if (param.kind == RuntimeParamKind::Shape) {
      ++contract.shape_param_count;
    } else {
      ++contract.scalar_param_count;
    }
  }
  return contract;
}

DispatchContract make_dispatch_contract(const StageRecord &stage) {
  DispatchContract contract;
  contract.execution_kind = stage.execution_kind;
  contract.backend_domain = stage.backend_domain;
  contract.kernel_unit_id = stage.kernel_unit_id;
  contract.kernel_unit_kind = stage.kernel_unit_kind;
  return contract;
}

MemoryContract make_memory_contract(const StageRecord &stage) {
  MemoryContract contract;
  contract.alias_group = "stage_" + std::to_string(stage.stage_id);
  return contract;
}

void verify_tensor_contract(const TensorContract &contract, const char *label,
                            size_t stage_id,
                            ManifestVerificationResult &result) {
  if (contract.logical_name.empty()) {
    result.diagnostics.push_back("stage " + std::to_string(stage_id) + " has " +
                                 label + " with empty logical name");
  }
  if (contract.memory_region_id.empty()) {
    result.diagnostics.push_back("stage " + std::to_string(stage_id) + " has " +
                                 label + " with empty memory region id");
  }
  if (contract.element_type.empty()) {
    result.diagnostics.push_back("stage " + std::to_string(stage_id) + " has " +
                                 label + " with empty element type");
  }
  if (contract.partial_shape.empty()) {
    result.diagnostics.push_back("stage " + std::to_string(stage_id) + " has " +
                                 label + " with empty partial shape");
  }
  if (contract.layout.empty() || contract.storage_kind.empty() ||
      contract.lifetime_class.empty()) {
    result.diagnostics.push_back("stage " + std::to_string(stage_id) + " has " +
                                 label + " with incomplete memory contract");
  }
}

void verify_runtime_param_contract(const RuntimeParamContract &contract,
                                   size_t stage_id,
                                   ManifestVerificationResult &result) {
  if (contract.scalar_param_count + contract.shape_param_count !=
          contract.params.size() ||
      contract.runtime_param_names.size() != contract.params.size()) {
    result.diagnostics.push_back("stage " + std::to_string(stage_id) +
                                 " has inconsistent runtime param counters");
  }
  for (size_t i = 0; i < contract.params.size(); ++i) {
    const auto &param = contract.params[i];
    if (param.logical_name.empty() || param.abi_type.empty() ||
        param.source_tensor.empty()) {
      result.diagnostics.push_back("stage " + std::to_string(stage_id) +
                                   " has incomplete runtime param descriptor");
    }
    if (i < contract.runtime_param_names.size() &&
        contract.runtime_param_names[i] != param.logical_name) {
      result.diagnostics.push_back("stage " + std::to_string(stage_id) +
                                   " has runtime param order drift");
    }
  }
}

} // namespace

ManifestVerificationResult ManifestBundle::verify() const {
  ManifestVerificationResult result;
  if (schema_version != kManifestSchemaVersion || target_fingerprint.empty()) {
    result.diagnostics.emplace_back("manifest header is incomplete");
  }
  const auto memory_result = memory_plan.verify();
  for (const auto &diagnostic : memory_result.diagnostics) {
    result.diagnostics.push_back("memory plan: " + diagnostic);
  }
  if (!stages.empty() && memory_plan.regions.empty()) {
    result.diagnostics.emplace_back("manifest memory plan is empty");
  }
  for (size_t i = 0; i < stages.size(); ++i) {
    const auto &stage = stages[i];
    if (stage.stage_id != i) {
      result.diagnostics.push_back("stage " + std::to_string(i) +
                                   " has non-contiguous stage id");
    }
    if (stage.stable_record_key == 0 || stage.normalized_op_family.empty() ||
        stage.backend_domain.empty() || stage.kernel_unit_id.empty() ||
        stage.kernel_unit_kind.empty()) {
      result.diagnostics.push_back("stage " + std::to_string(stage.stage_id) +
                                   " has incomplete identity");
    }
    if (stage.execution_kind == LoweringRouteKind::Unsupported) {
      result.diagnostics.push_back("stage " + std::to_string(stage.stage_id) +
                                   " has unsupported execution kind");
    }
    if (stage.execution_kind == LoweringRouteKind::HandwrittenKernelException &&
        !stage.handwritten_exception.valid()) {
      result.diagnostics.push_back(
          "stage " + std::to_string(stage.stage_id) +
          " has incomplete handwritten exception contract");
    }
    if (stage.execution_kind != LoweringRouteKind::HandwrittenKernelException &&
        (stage.handwritten_exception.ticket.empty() == false ||
         stage.handwritten_exception.reason.empty() == false ||
         stage.handwritten_exception.removal_condition.empty() == false)) {
      result.diagnostics.push_back(
          "stage " + std::to_string(stage.stage_id) +
          " carries handwritten exception metadata on a non-exception route");
    }
    if (stage.dispatch.execution_kind != stage.execution_kind ||
        stage.dispatch.backend_domain != stage.backend_domain ||
        stage.dispatch.kernel_unit_id != stage.kernel_unit_id ||
        stage.dispatch.kernel_unit_kind != stage.kernel_unit_kind ||
        stage.dispatch.dispatch_source.empty()) {
      result.diagnostics.push_back("stage " + std::to_string(stage.stage_id) +
                                   " has dispatch contract drift");
    }
    if (stage.memory.hidden_host_copy_allowed) {
      result.diagnostics.push_back("stage " + std::to_string(stage.stage_id) +
                                   " allows hidden host copies");
    }
    if (stage.memory.input_lifetime.empty() ||
        stage.memory.output_lifetime.empty() ||
        stage.memory.alias_group.empty()) {
      result.diagnostics.push_back("stage " + std::to_string(stage.stage_id) +
                                   " has incomplete memory contract");
    }
    if (!stage.memory.alias_group.empty() &&
        !memory_plan.has_alias_group(stage.memory.alias_group)) {
      result.diagnostics.push_back("stage " + std::to_string(stage.stage_id) +
                                   " memory alias group is missing from plan");
    }
    verify_runtime_param_contract(stage.runtime_params, stage.stage_id, result);
    for (const auto &input : stage.inputs) {
      verify_tensor_contract(input, "input", stage.stage_id, result);
      if (!input.memory_region_id.empty() &&
          !memory_plan.has_region(input.memory_region_id)) {
        result.diagnostics.push_back("stage " +
                                     std::to_string(stage.stage_id) +
                                     " input memory region is missing from plan");
      }
    }
    for (const auto &output : stage.outputs) {
      verify_tensor_contract(output, "output", stage.stage_id, result);
      if (!output.memory_region_id.empty() &&
          !memory_plan.has_region(output.memory_region_id)) {
        result.diagnostics.push_back("stage " +
                                     std::to_string(stage.stage_id) +
                                     " output memory region is missing from plan");
      }
    }
  }
  return result;
}

bool ManifestBundle::valid() const { return verify().valid(); }

size_t ManifestBundle::route_count(LoweringRouteKind route_kind) const {
  size_t count = 0;
  for (const auto &stage : stages) {
    if (stage.execution_kind == route_kind) {
      ++count;
    }
  }
  return count;
}

ManifestBundle ManifestBuilder::build(const LoweringPlan &plan) const {
  ManifestBundle manifest;
  manifest.schema_version = kManifestSchemaVersion;
  manifest.target_fingerprint = plan.target.fingerprint();
  manifest.memory_plan = MemoryPlanBuilder{}.build(plan);
  manifest.stages.reserve(plan.operations.size());
  for (size_t i = 0; i < plan.operations.size(); ++i) {
    const auto &op = plan.operations[i];
    StageRecord stage;
    stage.stage_id = i;
    stage.stable_record_key = stable_hash64(stage_key_material(plan, op, i));
    stage.source_node_name = op.node_name;
    stage.normalized_op_family = op.type_name;
    stage.execution_kind = op.kernel_unit.route_kind();
    stage.backend_domain = backend_domain(plan.target);
    stage.kernel_unit_id = op.kernel_unit.id();
    stage.kernel_unit_kind =
        std::string(kernel_unit_kind_to_string(op.kernel_unit.kind()));
    stage.requires_runtime_shape_args =
        op.kernel_unit.requires_runtime_shape_args();
    stage.inputs = make_input_contracts(op);
    stage.outputs = make_output_contracts(op);
    for (size_t input_idx = 0; input_idx < stage.inputs.size(); ++input_idx) {
      stage.inputs[input_idx].memory_region_id =
          "stage_" + std::to_string(i) + ".input_" + std::to_string(input_idx);
    }
    for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
      stage.outputs[output_idx].memory_region_id =
          "stage_" + std::to_string(i) + ".output_" + std::to_string(output_idx);
    }
    stage.runtime_params = make_runtime_param_contract(stage);
    stage.dispatch = make_dispatch_contract(stage);
    stage.memory = make_memory_contract(stage);
    stage.handwritten_exception = op.kernel_unit.exception_contract();
    stage.profitability_score = op.profitability_score;
    manifest.stages.push_back(std::move(stage));
  }
  return manifest;
}

std::string_view
tensor_contract_role_to_string(TensorContractRole role) noexcept {
  switch (role) {
  case TensorContractRole::TensorInput:
    return "tensor_input";
  case TensorContractRole::TensorOutput:
    return "tensor_output";
  }
  return "tensor_input";
}

std::string_view runtime_param_kind_to_string(RuntimeParamKind kind) noexcept {
  switch (kind) {
  case RuntimeParamKind::Scalar:
    return "scalar";
  case RuntimeParamKind::Shape:
    return "shape";
  }
  return "shape";
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
