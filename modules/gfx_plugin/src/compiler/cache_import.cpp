// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/cache_import.hpp"

#include <string>
#include <unordered_map>
#include <unordered_set>
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

PipelineStageTensorRef make_parameter_ref(size_t index) {
  PipelineStageTensorRef ref;
  ref.kind = PipelineStageTensorRefKind::Parameter;
  ref.index = index;
  ref.port = 0;
  return ref;
}

PipelineStageTensorRef make_stage_output_ref(size_t index, size_t port) {
  PipelineStageTensorRef ref;
  ref.kind = PipelineStageTensorRefKind::StageOutput;
  ref.index = index;
  ref.port = port;
  return ref;
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

void materialize_cache_pipeline(
    CacheImportContract &contract,
    RuntimeExecutableDescriptor &descriptor) {
  std::unordered_set<std::string> consumed_regions;
  for (const auto &stage : descriptor.stages) {
    for (const auto &input : stage.input_bindings) {
      if (!input.memory_region_id.empty()) {
        consumed_regions.insert(input.memory_region_id);
      }
    }
  }

  std::unordered_map<std::string, PipelineStageTensorRef> tensor_refs;
  std::unordered_map<std::string, size_t> parameter_index_by_region;
  std::vector<std::shared_ptr<ov::op::v0::Parameter>> parameters;
  std::vector<PublicOutputRecord> public_outputs;

  descriptor.materialization_stages.reserve(descriptor.stages.size());
  for (size_t stage_idx = 0; stage_idx < descriptor.stages.size();
       ++stage_idx) {
    const auto &stage = descriptor.stages[stage_idx];
    PipelineStageMaterializationPlan plan;
    plan.kind = PipelineStageMaterializationKind::SingleStage;
    plan.descriptor_stage_index = stage_idx;
    plan.materialized_descriptor = stage;
    plan.materialized_descriptor_valid = true;
    plan.io_plan.stage_name = stage.stage_name;
    plan.io_plan.op_family = stage.op_family;
    plan.io_plan.runtime_stage_index = stage_idx;

    for (size_t input_idx = 0; input_idx < stage.input_bindings.size();
         ++input_idx) {
      const auto &binding = stage.input_bindings[input_idx];
      PipelineStageInputLink input;
      input.port = input_idx;
      auto produced = tensor_refs.find(binding.memory_region_id);
      if (produced != tensor_refs.end()) {
        input.source_ref = produced->second;
      } else if (binding.external_binding) {
        auto param_it = parameter_index_by_region.find(binding.memory_region_id);
        if (param_it == parameter_index_by_region.end()) {
          const size_t parameter_index = parameters.size();
          parameter_index_by_region.emplace(binding.memory_region_id,
                                            parameter_index);
          parameters.push_back(make_parameter_for_binding(binding));
          input.source_ref = make_parameter_ref(parameter_index);
        } else {
          input.source_ref = make_parameter_ref(param_it->second);
        }
      }
      if (!input.source_ref.valid()) {
        contract.diagnostics.push_back(
            "cache import cannot resolve input binding " +
            std::to_string(stage_idx) + ":" + std::to_string(input_idx) +
            " region=" + binding.memory_region_id);
      }
      plan.io_plan.inputs.push_back(std::move(input));
    }

    for (size_t output_idx = 0; output_idx < stage.output_bindings.size();
         ++output_idx) {
      const auto &binding = stage.output_bindings[output_idx];
      PipelineStageOutputDesc output;
      output.source_port = output_idx;
      output.source_ref = make_stage_output_ref(stage_idx, output_idx);
      if (!static_shape_and_type_from_binding(
              binding, output.shape, output.type, contract.diagnostics,
              "cache stage output " + std::to_string(stage_idx) + ":" +
                  std::to_string(output_idx))) {
        continue;
      }
      output.is_model_output =
          consumed_regions.count(binding.memory_region_id) == 0u;
      plan.io_plan.outputs.push_back(output);
      tensor_refs[binding.memory_region_id] = output.source_ref;

      if (output.is_model_output) {
        RuntimePublicOutputDescriptor public_descriptor;
        public_descriptor.kind = RuntimePublicOutputSourceKind::StageOutput;
        public_descriptor.index = stage_idx;
        public_descriptor.port = output_idx;
        public_descriptor.static_shape = output.shape;
        public_descriptor.static_type = output.type;
        public_outputs.push_back({binding, public_descriptor});
      }
    }

    descriptor.materialization_stages.push_back(std::move(plan));
  }

  for (const auto &public_output : public_outputs) {
    descriptor.public_outputs.push_back(public_output.descriptor);
  }
  if (descriptor.public_outputs.empty() && !descriptor.stages.empty()) {
    contract.diagnostics.emplace_back(
        "cache import did not recover public model outputs");
  }

  descriptor.materialization_finalized = true;
  contract.runtime_model =
      make_runtime_model(parameters, public_outputs, contract.diagnostics);
  if (!contract.runtime_model) {
    contract.diagnostics.emplace_back(
        "cache import could not reconstruct OpenVINO runtime model contract");
  }
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

  auto descriptor =
      RuntimeExecutableDescriptorBuilder{}.build(contract.executable);
  const auto descriptor_verification =
      verify_runtime_executable_descriptor(descriptor, contract.executable);
  append_prefixed(contract.diagnostics, "runtime descriptor: ",
                  descriptor_verification.diagnostics);
  if (!contract.diagnostics.empty()) {
    return contract;
  }

  materialize_cache_pipeline(contract, descriptor);
  const auto materialization_verification =
      verify_runtime_executable_descriptor_materialization(descriptor);
  append_prefixed(contract.diagnostics, "runtime materialization: ",
                  materialization_verification.diagnostics);
  if (!contract.diagnostics.empty()) {
    return contract;
  }

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
