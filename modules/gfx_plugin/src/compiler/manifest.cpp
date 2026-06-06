// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/manifest.hpp"

#include <limits>
#include <sstream>
#include <utility>

#include "compiler/pipeline_stage_plan.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/util/assign_base.hpp"
#include "openvino/op/util/read_value_base.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

namespace {

constexpr uint32_t kManifestSchemaVersion = 2;
constexpr const char *kStatefulPrebindShapeRuleNone = "none";
constexpr const char *kStatefulPrebindShapeRuleStaticOutput =
    "static_output_shape";
constexpr const char *kStatefulPrebindShapeRuleSumInputsAlongAxis =
    "sum_inputs_along_axis";
constexpr const char *kRuntimeShapeRuleStaticOrDescriptor =
    "static_or_descriptor";
constexpr const char *kRuntimeShapeRuleConcat = "concat";
constexpr const char *kRuntimeShapeRuleBroadcast = "broadcast";
constexpr const char *kRuntimeShapeRuleSelect = "select";
constexpr const char *kRuntimeShapeRuleShapeOf = "shape_of";
constexpr const char *kRuntimeShapeRuleSlice = "slice";
constexpr const char *kRuntimeShapeRuleRange = "range";
constexpr const char *kRuntimeShapeRuleTile = "tile";
constexpr int64_t kRuntimeSliceMetadataVersion = 1;
constexpr int64_t kRuntimeSliceKindV8 = 1;
constexpr int64_t kRuntimeSliceKindStridedSliceV1 = 2;

uint64_t stable_hash64(std::string_view value) noexcept {
  uint64_t hash = 14695981039346656037ull;
  for (const unsigned char c : value) {
    hash ^= c;
    hash *= 1099511628211ull;
  }
  return hash;
}

StatefulEffectContract make_stateful_effect_contract(
    const PlannedOperation &op) {
  StatefulEffectContract contract;
  if (auto read =
          ov::as_type_ptr<const ov::op::util::ReadValueBase>(op.source_node)) {
    contract.kind = StatefulEffectKind::ReadValue;
    contract.variable_id = read->get_variable_id();
    return contract;
  }
  if (auto assign =
          ov::as_type_ptr<const ov::op::util::AssignBase>(op.source_node)) {
    contract.kind = StatefulEffectKind::Assign;
    contract.variable_id = assign->get_variable_id();
    return contract;
  }
  return contract;
}

std::string stage_key_material(const LoweringPlan &plan,
                               const PlannedOperation &op, size_t stage_id,
                               const StatefulEffectContract &stateful_effect) {
  std::ostringstream os;
  os << plan.target.fingerprint() << "#" << stage_id << "#" << op.node_name
     << "#" << op.type_name << "#" << op.kernel_unit.manifest_key() << "#"
     << stateful_effect_kind_to_string(stateful_effect.kind) << "#"
     << stateful_effect.variable_id;
  return os.str();
}

std::string backend_domain(const BackendTarget &target) {
  return target.backend_id();
}

bool shape_is_dynamic(const std::string &partial_shape) {
  return partial_shape.find('?') != std::string::npos ||
         partial_shape.find("-1") != std::string::npos;
}

bool stateful_prebind_shape_rule_valid(std::string_view rule) {
  return rule == kStatefulPrebindShapeRuleNone ||
         rule == kStatefulPrebindShapeRuleStaticOutput ||
         rule == kStatefulPrebindShapeRuleSumInputsAlongAxis;
}

bool runtime_shape_rule_valid(std::string_view rule) {
  return rule == kRuntimeShapeRuleStaticOrDescriptor ||
         rule == kRuntimeShapeRuleConcat || rule == kRuntimeShapeRuleBroadcast ||
         rule == kRuntimeShapeRuleSelect || rule == kRuntimeShapeRuleShapeOf ||
         rule == kRuntimeShapeRuleSlice || rule == kRuntimeShapeRuleRange ||
         rule == kRuntimeShapeRuleTile;
}

bool read_counted_metadata_field(const std::vector<int64_t> &metadata,
                                 size_t &offset) {
  if (offset >= metadata.size() || metadata[offset] < 0) {
    return false;
  }
  const size_t count = static_cast<size_t>(metadata[offset++]);
  if (offset + count > metadata.size()) {
    return false;
  }
  offset += count;
  return true;
}

bool slice_runtime_shape_metadata_valid(const std::vector<int64_t> &metadata) {
  if (metadata.size() < 3 ||
      metadata[0] != kRuntimeSliceMetadataVersion ||
      (metadata[1] != kRuntimeSliceKindV8 &&
       metadata[1] != kRuntimeSliceKindStridedSliceV1) ||
      metadata[2] < 3) {
    return false;
  }
  if (metadata[1] == kRuntimeSliceKindV8) {
    return metadata.size() == 3;
  }
  size_t offset = 3;
  for (size_t i = 0; i < 5; ++i) {
    if (!read_counted_metadata_field(metadata, offset)) {
      return false;
    }
  }
  return offset == metadata.size();
}

std::string runtime_shape_rule_for_op(const PlannedOperation &op) {
  if (op.type_name == "Concat") {
    return kRuntimeShapeRuleConcat;
  }
  if (op.type_name == "Broadcast") {
    return kRuntimeShapeRuleBroadcast;
  }
  if (op.type_name == "Select") {
    return kRuntimeShapeRuleSelect;
  }
  if (op.type_name == "ShapeOf") {
    return kRuntimeShapeRuleShapeOf;
  }
  if (op.type_name == "Slice" || op.type_name == "StridedSlice") {
    return kRuntimeShapeRuleSlice;
  }
  if (op.type_name == "Range") {
    return kRuntimeShapeRuleRange;
  }
  if (op.type_name == "Tile") {
    return kRuntimeShapeRuleTile;
  }
  return kRuntimeShapeRuleStaticOrDescriptor;
}

void append_counted_i64_vector(std::vector<int64_t> &metadata,
                               const std::vector<int64_t> &values) {
  metadata.push_back(static_cast<int64_t>(values.size()));
  metadata.insert(metadata.end(), values.begin(), values.end());
}

RuntimeShapeContract make_slice_runtime_shape_contract(
    const ov::op::v8::Slice &slice) {
  RuntimeShapeContract contract;
  contract.rule = kRuntimeShapeRuleSlice;
  contract.i64_metadata = {kRuntimeSliceMetadataVersion, kRuntimeSliceKindV8,
                           static_cast<int64_t>(slice.get_input_size())};
  return contract;
}

RuntimeShapeContract make_strided_slice_runtime_shape_contract(
    const ov::op::v1::StridedSlice &slice) {
  RuntimeShapeContract contract;
  contract.rule = kRuntimeShapeRuleSlice;
  contract.i64_metadata = {kRuntimeSliceMetadataVersion,
                           kRuntimeSliceKindStridedSliceV1,
                           static_cast<int64_t>(slice.get_input_size())};
  append_counted_i64_vector(contract.i64_metadata, slice.get_begin_mask());
  append_counted_i64_vector(contract.i64_metadata, slice.get_end_mask());
  append_counted_i64_vector(contract.i64_metadata, slice.get_new_axis_mask());
  append_counted_i64_vector(contract.i64_metadata, slice.get_shrink_axis_mask());
  append_counted_i64_vector(contract.i64_metadata, slice.get_ellipsis_mask());
  return contract;
}

RuntimeShapeContract runtime_shape_contract_for_op(const PlannedOperation &op) {
  RuntimeShapeContract contract;
  contract.rule = runtime_shape_rule_for_op(op);

  if (auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(op.source_node)) {
    contract.i64_metadata.push_back(concat->get_axis());
    return contract;
  }

  if (auto broadcast =
          ov::as_type_ptr<const ov::op::v3::Broadcast>(op.source_node)) {
    const bool bidirectional =
        broadcast->get_broadcast_spec().m_type ==
        ov::op::BroadcastType::BIDIRECTIONAL;
    contract.i64_metadata.push_back(bidirectional ? 1 : 0);
    const auto rank = broadcast->get_output_partial_shape(0).rank();
    contract.i64_metadata.push_back(
        rank.is_static() ? rank.get_length() : -1);
    return contract;
  }

  if (auto broadcast =
          ov::as_type_ptr<const ov::op::v1::Broadcast>(op.source_node)) {
    contract.i64_metadata.push_back(
        broadcast->get_broadcast_spec().m_type ==
                ov::op::AutoBroadcastType::NUMPY
            ? 1
            : 0);
    const auto rank = broadcast->get_output_partial_shape(0).rank();
    contract.i64_metadata.push_back(
        rank.is_static() ? rank.get_length() : -1);
    return contract;
  }

  if (auto slice = ov::as_type_ptr<const ov::op::v8::Slice>(op.source_node)) {
    return make_slice_runtime_shape_contract(*slice);
  }

  if (auto slice =
          ov::as_type_ptr<const ov::op::v1::StridedSlice>(op.source_node)) {
    return make_strided_slice_runtime_shape_contract(*slice);
  }

  return contract;
}

int64_t normalize_axis_for_rank(int64_t axis, size_t rank) {
  if (axis < 0) {
    axis += static_cast<int64_t>(rank);
  }
  return axis >= 0 && static_cast<size_t>(axis) < rank ? axis : -1;
}

std::string stateful_prebind_shape_rule(const PlannedOperation &op,
                                        size_t output_idx,
                                        int64_t &shape_axis) {
  shape_axis = -1;
  if (output_idx >= op.output_shapes.size()) {
    return kStatefulPrebindShapeRuleNone;
  }
  if (!shape_is_dynamic(op.output_shapes[output_idx])) {
    return kStatefulPrebindShapeRuleStaticOutput;
  }
  auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(op.source_node);
  if (!concat || output_idx >= concat->get_output_size()) {
    return kStatefulPrebindShapeRuleNone;
  }
  const auto rank = concat->get_output_partial_shape(output_idx).rank();
  if (!rank.is_static()) {
    return kStatefulPrebindShapeRuleNone;
  }
  shape_axis =
      normalize_axis_for_rank(concat->get_axis(), static_cast<size_t>(rank.get_length()));
  return shape_axis >= 0 ? kStatefulPrebindShapeRuleSumInputsAlongAxis
                         : kStatefulPrebindShapeRuleNone;
}

void annotate_stateful_prebind_contract(const PlannedOperation &op,
                                        size_t output_idx,
                                        TensorContract &contract) {
  const auto variable_id =
      direct_stateful_assign_variable_id(op.source_node, output_idx);
  if (variable_id.empty()) {
    return;
  }
  contract.stateful_prebind_variable_id = variable_id;
  contract.stateful_prebind_shape_rule =
      stateful_prebind_shape_rule(op, output_idx,
                                  contract.stateful_prebind_shape_axis);
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
    auto contract = make_tensor_contract(op, TensorContractRole::TensorOutput,
                                         i, op.output_element_types[i], shape);
    annotate_stateful_prebind_contract(op, i, contract);
    contracts.push_back(std::move(contract));
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

uint64_t mul_saturating(uint64_t lhs, uint64_t rhs) {
  if (lhs == 0 || rhs == 0) {
    return 0;
  }
  if (lhs > std::numeric_limits<uint64_t>::max() / rhs) {
    return std::numeric_limits<uint64_t>::max();
  }
  return lhs * rhs;
}

uint64_t safe_shape_size_u64(const ov::Shape &shape) {
  uint64_t size = 1;
  for (const auto dim : shape) {
    size = mul_saturating(size, static_cast<uint64_t>(dim));
  }
  return size;
}

uint64_t convolution_macs_estimate(const ov::op::v1::Convolution &conv) {
  if (!conv.get_input_partial_shape(1).is_static() ||
      !conv.get_output_partial_shape(0).is_static()) {
    return 0;
  }
  const auto weights = conv.get_input_shape(1);
  const auto output = conv.get_output_shape(0);
  if (weights.size() != 4 || output.size() != 4) {
    return 0;
  }
  const uint64_t reduction =
      mul_saturating(static_cast<uint64_t>(weights[1]),
                     mul_saturating(static_cast<uint64_t>(weights[2]),
                                    static_cast<uint64_t>(weights[3])));
  return mul_saturating(safe_shape_size_u64(output), reduction);
}

uint64_t group_convolution_macs_estimate(
    const ov::op::v1::GroupConvolution &group_conv) {
  if (!group_conv.get_input_partial_shape(1).is_static() ||
      !group_conv.get_output_partial_shape(0).is_static()) {
    return 0;
  }
  const auto weights = group_conv.get_input_shape(1);
  const auto output = group_conv.get_output_shape(0);
  if (weights.size() != 5 || output.size() != 4) {
    return 0;
  }
  const uint64_t reduction =
      mul_saturating(static_cast<uint64_t>(weights[2]),
                     mul_saturating(static_cast<uint64_t>(weights[3]),
                                    static_cast<uint64_t>(weights[4])));
  return mul_saturating(safe_shape_size_u64(output), reduction);
}

uint64_t matmul_macs_estimate(const ov::op::v0::MatMul &matmul) {
  if (!matmul.get_input_partial_shape(0).is_static() ||
      !matmul.get_input_partial_shape(1).is_static() ||
      !matmul.get_output_partial_shape(0).is_static()) {
    return 0;
  }
  const auto a = matmul.get_input_shape(0);
  const auto b = matmul.get_input_shape(1);
  const auto output = matmul.get_output_shape(0);
  if (a.size() < 2 || b.size() < 2 || output.size() < 2) {
    return 0;
  }
  const auto k = matmul.get_transpose_a() ? a[a.size() - 2] : a[a.size() - 1];
  return mul_saturating(safe_shape_size_u64(output), static_cast<uint64_t>(k));
}

uint64_t workload_macs_estimate(const PlannedOperation &op) {
  if (!op.source_node) {
    return 0;
  }
  try {
    if (auto conv =
            ov::as_type_ptr<const ov::op::v1::Convolution>(op.source_node)) {
      return convolution_macs_estimate(*conv);
    }
    if (auto group_conv =
            ov::as_type_ptr<const ov::op::v1::GroupConvolution>(
                op.source_node)) {
      return group_convolution_macs_estimate(*group_conv);
    }
    if (auto matmul =
            ov::as_type_ptr<const ov::op::v0::MatMul>(op.source_node)) {
      return matmul_macs_estimate(*matmul);
    }
  } catch (const std::exception &) {
    return 0;
  }
  return 0;
}

bool dependency_extension_boundary_for_op(std::string_view type_name) {
  return type_name == "Concat" || type_name == "Split" ||
         type_name == "VariadicSplit" || type_name == "Transpose" ||
         type_name == "Reshape" || type_name == "Squeeze" ||
         type_name == "Unsqueeze" || type_name == "Softmax" ||
         type_name == "LogSoftmax" || type_name == "FusedAttention";
}

SubmissionContract make_submission_contract(const PlannedOperation &op) {
  SubmissionContract contract;
  contract.stage_weight = 1;
  contract.macs_estimate = workload_macs_estimate(op);
  contract.dependency_extension_boundary =
      dependency_extension_boundary_for_op(op.type_name);
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
  if (contract.stateful_prebind_shape_rule.empty() ||
      !stateful_prebind_shape_rule_valid(
          contract.stateful_prebind_shape_rule)) {
    result.diagnostics.push_back("stage " + std::to_string(stage_id) + " has " +
                                 label +
                                 " with invalid stateful prebind shape rule");
  }
  if (contract.stateful_prebind_variable_id.empty() &&
      contract.stateful_prebind_shape_rule != kStatefulPrebindShapeRuleNone) {
    result.diagnostics.push_back("stage " + std::to_string(stage_id) + " has " +
                                 label +
                                 " with stateful prebind shape rule but no "
                                 "variable id");
  }
  if (contract.stateful_prebind_shape_rule ==
          kStatefulPrebindShapeRuleSumInputsAlongAxis &&
      contract.stateful_prebind_shape_axis < 0) {
    result.diagnostics.push_back("stage " + std::to_string(stage_id) + " has " +
                                 label +
                                 " with stateful prebind axis missing");
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

void verify_runtime_shape_contract(const RuntimeShapeContract &contract,
                                   size_t stage_id,
                                   ManifestVerificationResult &result) {
  if (contract.rule.empty() || !runtime_shape_rule_valid(contract.rule)) {
    result.diagnostics.push_back("stage " + std::to_string(stage_id) +
                                 " has invalid runtime shape contract");
  }
  if (contract.rule == kRuntimeShapeRuleConcat &&
      contract.i64_metadata.size() != 1) {
    result.diagnostics.push_back("stage " + std::to_string(stage_id) +
                                 " has incomplete concat runtime shape "
                                 "metadata");
  }
  if (contract.rule == kRuntimeShapeRuleBroadcast &&
      contract.i64_metadata.size() < 2) {
    result.diagnostics.push_back("stage " + std::to_string(stage_id) +
                                 " has incomplete broadcast runtime shape "
                                 "metadata");
  }
  if (contract.rule == kRuntimeShapeRuleSlice) {
    if (!slice_runtime_shape_metadata_valid(contract.i64_metadata)) {
      result.diagnostics.push_back("stage " + std::to_string(stage_id) +
                                   " has incomplete slice runtime shape "
                                   "metadata");
    }
  }
}

void verify_submission_contract(const SubmissionContract &contract,
                                size_t stage_id,
                                ManifestVerificationResult &result) {
  if (contract.stage_weight == 0) {
    result.diagnostics.push_back("stage " + std::to_string(stage_id) +
                                 " has zero submission stage weight");
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
    verify_runtime_shape_contract(stage.runtime_shape, stage.stage_id, result);
    verify_submission_contract(stage.submission, stage.stage_id, result);
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
    const auto stateful_effect = make_stateful_effect_contract(op);
    StageRecord stage;
    stage.stage_id = i;
    stage.stable_record_key =
        stable_hash64(stage_key_material(plan, op, i, stateful_effect));
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
    stage.runtime_shape = runtime_shape_contract_for_op(op);
    stage.stateful_effect = std::move(stateful_effect);
    stage.dispatch = make_dispatch_contract(stage);
    stage.memory = make_memory_contract(stage);
    stage.submission = make_submission_contract(op);
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

std::string_view
stateful_effect_kind_to_string(StatefulEffectKind kind) noexcept {
  switch (kind) {
  case StatefulEffectKind::None:
    return "none";
  case StatefulEffectKind::ReadValue:
    return "read_value";
  case StatefulEffectKind::Assign:
    return "assign";
  }
  return "none";
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
