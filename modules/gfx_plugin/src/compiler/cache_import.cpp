// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/cache_import.hpp"

#include <map>
#include <string>
#include <unordered_map>
#include <utility>

#include "compiler/runtime_executable_descriptor_builder.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "runtime/tensor_binding_contract.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

void append_prefixed(std::vector<std::string> &dst, std::string prefix,
                     const std::vector<std::string> &src) {
  for (const auto &diagnostic : src) {
    dst.push_back(prefix + diagnostic);
  }
}

BackendTarget resolve_cache_target(const CacheEnvelope &envelope,
                                   const BackendRegistry &registry,
                                   std::vector<std::string> &diagnostics) {
  const auto targets = registry.available_targets();
  for (const auto &target : targets) {
    if (target.is_compatible_with_fingerprint(envelope.key.target_fingerprint)) {
      return target;
    }
  }
  diagnostics.push_back("cache envelope target is not available on this host: " +
                        envelope.key.target_fingerprint);
  return {};
}

ov::PartialShape partial_shape_from_contract(std::string_view contract) {
  ov::Shape shape;
  if (parse_static_shape_contract(contract, shape)) {
    return ov::PartialShape(shape);
  }
  return ov::PartialShape::dynamic();
}

bool static_shape_and_type_from_binding(
    const RuntimeTensorBindingContract &binding, ov::Shape &shape,
    ov::element::Type &type, std::vector<std::string> &diagnostics,
    std::string label) {
  type = element_type_from_contract(binding.element_type);
  if (type == ov::element::dynamic) {
    diagnostics.push_back(label + " has dynamic or unsupported element type");
    return false;
  }
  if (!parse_static_shape_contract(binding.partial_shape, shape)) {
    diagnostics.push_back(label + " has dynamic or unsupported shape");
    return false;
  }
  return true;
}

std::shared_ptr<ov::op::v0::Parameter>
make_parameter_for_binding(const RuntimeTensorBindingContract &binding) {
  const auto type = element_type_from_contract(binding.element_type);
  auto parameter = std::make_shared<ov::op::v0::Parameter>(
      type, partial_shape_from_contract(binding.partial_shape));
  parameter->set_friendly_name(binding.logical_name.empty()
                                   ? binding.memory_region_id
                                   : binding.logical_name);
  if (!binding.logical_name.empty()) {
    parameter->output(0).get_tensor().set_names({binding.logical_name});
  }
  return parameter;
}

struct PublicOutputRecord {
  RuntimeTensorBindingContract binding;
  RuntimePublicOutputDescriptor descriptor;
};

std::shared_ptr<const ov::Model> make_runtime_model(
    const std::vector<std::shared_ptr<ov::op::v0::Parameter>> &parameters,
    const std::vector<PublicOutputRecord> &public_outputs,
    std::vector<std::string> &diagnostics) {
  ov::ResultVector results;
  results.reserve(public_outputs.size());
  for (size_t i = 0; i < public_outputs.size(); ++i) {
    ov::Shape shape;
    ov::element::Type type;
    if (!static_shape_and_type_from_binding(
            public_outputs[i].binding, shape, type, diagnostics,
            "cache public output " + std::to_string(i))) {
      continue;
    }
    auto source = std::make_shared<ov::op::v0::Constant>(type, shape);
    auto result = std::make_shared<ov::op::v0::Result>(source);
    result->set_friendly_name(public_outputs[i].binding.logical_name.empty()
                                  ? "output" + std::to_string(i)
                                  : public_outputs[i].binding.logical_name);
    results.push_back(std::move(result));
  }
  if (results.size() != public_outputs.size()) {
    return nullptr;
  }
  ov::ParameterVector model_parameters(parameters.begin(), parameters.end());
  auto model = std::make_shared<ov::Model>(results, model_parameters,
                                           "gfx_cache_imported_model");
  return model;
}

std::string shape_to_contract(const ov::Shape &shape) {
  std::string result = "{";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) {
      result += ",";
    }
    result += std::to_string(shape[i]);
  }
  result += "}";
  return result;
}

bool same_binding_identity(const RuntimeTensorBindingContract &lhs,
                           const RuntimeTensorBindingContract &rhs) {
  return lhs.memory_region_id == rhs.memory_region_id &&
         lhs.logical_name == rhs.logical_name &&
         lhs.element_type == rhs.element_type &&
         lhs.partial_shape == rhs.partial_shape &&
         lhs.layout == rhs.layout && lhs.storage_kind == rhs.storage_kind;
}

RuntimeTensorBindingContract make_public_output_binding(
    const RuntimePublicOutputDescriptor &output, size_t output_index) {
  RuntimeTensorBindingContract binding;
  binding.logical_name = "output" + std::to_string(output_index);
  binding.memory_region_id = binding.logical_name;
  binding.role = "tensor_output";
  binding.element_type = output.static_type.get_type_name();
  binding.partial_shape = shape_to_contract(output.static_shape);
  binding.layout = "logical";
  binding.storage_kind = "device_buffer";
  binding.lifetime_class = "stage_output";
  binding.alias_group = binding.memory_region_id;
  return binding;
}

struct CacheRuntimeModelInputs {
  std::vector<std::shared_ptr<ov::op::v0::Parameter>> parameters;
  std::unordered_map<size_t, RuntimeTensorBindingContract> bindings_by_index;
};

CacheRuntimeModelInputs collect_runtime_model_inputs(
    const RuntimeExecutableDescriptor &descriptor,
    std::vector<std::string> &diagnostics) {
  std::map<size_t, RuntimeTensorBindingContract> ordered_bindings;
  for (const auto &plan : descriptor.materialization_stages) {
    const auto &stage = plan.materialized_descriptor;
    for (const auto &input : plan.io_plan.inputs) {
      if (input.source_ref.kind != PipelineStageTensorRefKind::Parameter) {
        continue;
      }
      if (input.port >= stage.input_bindings.size()) {
        diagnostics.push_back(
            "cache import materialization parameter binding out of range: " +
            std::to_string(input.source_ref.index) + ":" +
            std::to_string(input.port));
        continue;
      }
      const auto &binding = stage.input_bindings[input.port];
      const auto [it, inserted] =
          ordered_bindings.emplace(input.source_ref.index, binding);
      if (!inserted && !same_binding_identity(it->second, binding)) {
        diagnostics.push_back(
            "cache import materialization parameter binding drift at " +
            std::to_string(input.source_ref.index));
      }
    }
  }

  CacheRuntimeModelInputs inputs;
  inputs.bindings_by_index.reserve(ordered_bindings.size());
  inputs.parameters.reserve(ordered_bindings.size());
  size_t expected_index = 0;
  for (const auto &[index, binding] : ordered_bindings) {
    if (index != expected_index) {
      diagnostics.push_back(
          "cache import materialization parameter index gap before " +
          std::to_string(index));
    }
    inputs.bindings_by_index.emplace(index, binding);
    inputs.parameters.push_back(make_parameter_for_binding(binding));
    expected_index = index + 1;
  }
  return inputs;
}

std::vector<PublicOutputRecord> collect_runtime_model_outputs(
    const RuntimeExecutableDescriptor &descriptor,
    const CacheRuntimeModelInputs &inputs,
    std::vector<std::string> &diagnostics) {
  std::vector<PublicOutputRecord> outputs;
  outputs.reserve(descriptor.public_outputs.size());
  for (size_t i = 0; i < descriptor.public_outputs.size(); ++i) {
    const auto &public_output = descriptor.public_outputs[i];
    RuntimeTensorBindingContract binding =
        make_public_output_binding(public_output, i);
    if (public_output.kind == RuntimePublicOutputSourceKind::Parameter) {
      const auto binding_it =
          inputs.bindings_by_index.find(public_output.index);
      if (binding_it == inputs.bindings_by_index.end()) {
        diagnostics.push_back(
            "cache import public output parameter is out of range at " +
            std::to_string(i));
        continue;
      }
      binding = binding_it->second;
    } else if (public_output.kind ==
               RuntimePublicOutputSourceKind::StageOutput) {
      if (public_output.index >= descriptor.materialization_stages.size()) {
        diagnostics.push_back(
            "cache import public output stage is out of range at " +
            std::to_string(i));
        continue;
      }
      const auto &plan =
          descriptor.materialization_stages[public_output.index];
      if (public_output.port <
          plan.materialized_descriptor.output_bindings.size()) {
        binding =
            plan.materialized_descriptor.output_bindings[public_output.port];
      }
    } else {
      diagnostics.push_back("cache import public output kind is incomplete at " +
                            std::to_string(i));
      continue;
    }
    outputs.push_back({std::move(binding), public_output});
  }
  return outputs;
}

std::shared_ptr<const ov::Model> make_runtime_model_from_materialization(
    const RuntimeExecutableDescriptor &descriptor,
    std::vector<std::string> &diagnostics) {
  const auto inputs = collect_runtime_model_inputs(descriptor, diagnostics);
  const auto outputs =
      collect_runtime_model_outputs(descriptor, inputs, diagnostics);
  if (outputs.empty() && !descriptor.stages.empty()) {
    diagnostics.emplace_back(
        "cache import materialization contract has no public outputs");
  }
  auto model = make_runtime_model(inputs.parameters, outputs, diagnostics);
  if (!model) {
    diagnostics.emplace_back(
        "cache import could not reconstruct OpenVINO runtime model contract");
  }
  return model;
}

} // namespace

CacheImportContract
make_cache_import_contract(const CacheEnvelope &envelope,
                           const BackendRegistry &registry) {
  CacheImportContract contract;
  contract.target = resolve_cache_target(envelope, registry,
                                         contract.diagnostics);
  auto backend_module = registry.resolve(contract.target);
  if (!backend_module) {
    contract.diagnostics.push_back(
        "cache envelope backend module is not available for target: " +
        contract.target.debug_string());
    return contract;
  }
  contract.executable = make_cache_envelope_executable_contract(
      envelope,
      [backend_module](const CacheBackendPayloadRecord &payload,
                       const KernelArtifactDescriptor &descriptor) {
        return backend_module->decode_cache_payload(payload, descriptor);
      });
  const auto executable_verification = contract.executable.verify();
  append_prefixed(contract.diagnostics, "executable: ",
                  executable_verification.diagnostics);
  const auto envelope_verification = envelope.verify(contract.executable);
  append_prefixed(contract.diagnostics, "cache envelope: ",
                  envelope_verification.diagnostics);
  if (!contract.diagnostics.empty()) {
    return contract;
  }

  auto descriptor = make_cache_envelope_runtime_descriptor_contract(
      envelope, contract.executable);
  const auto descriptor_verification =
      verify_runtime_executable_descriptor(descriptor, contract.executable);
  append_prefixed(contract.diagnostics, "runtime descriptor: ",
                  descriptor_verification.diagnostics);
  const auto materialization_verification =
      verify_runtime_executable_descriptor_materialization(descriptor);
  append_prefixed(contract.diagnostics, "runtime materialization: ",
                  materialization_verification.diagnostics);
  if (!contract.diagnostics.empty()) {
    return contract;
  }

  contract.runtime_model =
      make_runtime_model_from_materialization(descriptor,
                                              contract.diagnostics);
  contract.runtime_descriptor =
      std::make_shared<RuntimeExecutableDescriptor>(std::move(descriptor));
  if (!contract.runtime_descriptor || !contract.runtime_model) {
    contract.diagnostics.emplace_back(
        "cache import did not produce complete compiled-model contract");
  }
  return contract;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
