// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "compiler/executable_bundle.hpp"
#include "compiler/runtime_executable_descriptor_builder.hpp"
#include "runtime/tensor_binding_contract.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

PipelineStageTensorRef make_cache_test_parameter_ref(size_t index) {
  PipelineStageTensorRef ref;
  ref.kind = PipelineStageTensorRefKind::Parameter;
  ref.index = index;
  ref.port = 0;
  return ref;
}

PipelineStageTensorRef make_cache_test_stage_output_ref(size_t index,
                                                        size_t port) {
  PipelineStageTensorRef ref;
  ref.kind = PipelineStageTensorRefKind::StageOutput;
  ref.index = index;
  ref.port = port;
  return ref;
}

PipelineStageOutputDesc make_cache_test_output_desc(
    const RuntimeTensorBindingContract &binding, size_t stage_index,
    size_t output_index, bool model_output) {
  PipelineStageOutputDesc output;
  output.source_port = output_index;
  output.source_ref = make_cache_test_stage_output_ref(stage_index,
                                                       output_index);
  output.is_model_output = model_output;
  parse_static_shape_contract(binding.partial_shape, output.shape);
  output.type = element_type_from_contract(binding.element_type);
  return output;
}

RuntimePublicOutputDescriptor make_cache_test_public_output(
    const PipelineStageOutputDesc &output, size_t materialization_stage_index) {
  RuntimePublicOutputDescriptor descriptor;
  descriptor.kind = RuntimePublicOutputSourceKind::StageOutput;
  descriptor.index = materialization_stage_index;
  descriptor.port = output.source_port;
  descriptor.static_shape = output.shape;
  descriptor.static_type = output.type;
  return descriptor;
}

RuntimeExecutableDescriptor make_finalized_cache_test_runtime_descriptor(
    const compiler::ExecutableBundle &executable) {
  auto descriptor =
      compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);

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
      const auto produced = tensor_refs.find(binding.memory_region_id);
      if (produced != tensor_refs.end()) {
        input.source_ref = produced->second;
      } else {
        auto [param_it, inserted] = parameter_index_by_region.emplace(
            binding.memory_region_id, parameter_index_by_region.size());
        (void)inserted;
        input.source_ref = make_cache_test_parameter_ref(param_it->second);
      }
      plan.io_plan.inputs.push_back(input);
    }

    for (size_t output_idx = 0; output_idx < stage.output_bindings.size();
         ++output_idx) {
      const auto &binding = stage.output_bindings[output_idx];
      const bool model_output =
          consumed_regions.count(binding.memory_region_id) == 0u;
      auto output = make_cache_test_output_desc(binding, stage_idx,
                                                output_idx, model_output);
      tensor_refs[binding.memory_region_id] = output.source_ref;
      if (model_output) {
        descriptor.public_outputs.push_back(make_cache_test_public_output(
            output, descriptor.materialization_stages.size()));
      }
      plan.io_plan.outputs.push_back(std::move(output));
    }

    descriptor.materialization_stages.push_back(std::move(plan));
  }

  descriptor.materialization_finalized = true;
  return descriptor;
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
