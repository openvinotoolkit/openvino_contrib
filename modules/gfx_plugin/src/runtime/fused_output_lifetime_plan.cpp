// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/fused_output_lifetime_plan.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace ov {
namespace gfx_plugin {
namespace {

const RuntimeMemoryAliasGroupDescriptor *find_alias_group(
    const RuntimeMemoryPlanDescriptor &memory_plan, const std::string &group_id) {
  if (group_id.empty()) {
    return nullptr;
  }
  const auto it = std::find_if(
      memory_plan.alias_groups.begin(), memory_plan.alias_groups.end(),
      [&](const RuntimeMemoryAliasGroupDescriptor &group) {
        return group.group_id == group_id;
      });
  return it == memory_plan.alias_groups.end() ? nullptr : &*it;
}

bool descriptor_outputs_may_alias_inputs(
    const RuntimeStageExecutableDescriptor *descriptor,
    const RuntimeMemoryPlanDescriptor &memory_plan) {
  if (!descriptor) {
    return false;
  }
  if (descriptor->tensor_view_only) {
    return true;
  }
  for (const auto &output : descriptor->output_bindings) {
    const auto *alias_group = find_alias_group(memory_plan, output.alias_group);
    if (alias_group && alias_group->output_aliasing) {
      return true;
    }
  }
  return false;
}

bool descriptor_outputs_share_first_input_storage(
    const RuntimeStageExecutableDescriptor *descriptor) {
  if (!descriptor || descriptor->input_bindings.empty()) {
    return false;
  }
  const auto &first_input_alias_group =
      descriptor->input_bindings.front().alias_group;
  if (first_input_alias_group.empty()) {
    return false;
  }
  return std::any_of(descriptor->output_bindings.begin(),
                     descriptor->output_bindings.end(),
                     [&](const RuntimeTensorBindingContract &output) {
                       return output.alias_group == first_input_alias_group;
                     });
}

}  // namespace

std::vector<RuntimeOutputLifetime> build_fused_output_lifetime_plan(
    const std::vector<FusedOutputLifetimeStage> &stages,
    const RuntimeMemoryPlanDescriptor &memory_plan, size_t output_count) {
  if (output_count == 0 || stages.empty()) {
    return {};
  }

  std::vector<RuntimeOutputLifetime> lifetimes(output_count);
  for (size_t stage_idx = 0; stage_idx < stages.size(); ++stage_idx) {
    const auto &stage = stages[stage_idx];
    for (const auto output_index : stage.output_indices) {
      if (output_index >= lifetimes.size()) {
        continue;
      }
      auto &lifetime = lifetimes[output_index];
      lifetime.produced_at = std::min(lifetime.produced_at, stage_idx);
      lifetime.last_used_at =
          std::max(lifetime.last_used_at == RuntimeOutputLifetime::npos
                       ? stage_idx
                       : lifetime.last_used_at,
                   stage_idx);
    }
    for (const auto &input : stage.inputs) {
      if (input.kind != FusedOutputLifetimeInputRef::Kind::Output ||
          input.index >= lifetimes.size()) {
        continue;
      }
      auto &lifetime = lifetimes[input.index];
      if (!lifetime.valid()) {
        continue;
      }
      lifetime.last_used_at = std::max(lifetime.last_used_at, stage_idx);
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t stage_idx = 0; stage_idx < stages.size(); ++stage_idx) {
      const auto &stage = stages[stage_idx];
      if (!descriptor_outputs_may_alias_inputs(stage.descriptor, memory_plan)) {
        continue;
      }
      if (stage.inputs.empty()) {
        continue;
      }
      const auto &input = stage.inputs.front();
      if (input.kind != FusedOutputLifetimeInputRef::Kind::Output ||
          input.index >= lifetimes.size()) {
        continue;
      }
      auto &input_lifetime = lifetimes[input.index];
      if (!input_lifetime.valid()) {
        continue;
      }
      for (const auto output_index : stage.output_indices) {
        if (output_index >= lifetimes.size()) {
          continue;
        }
        const auto &output_lifetime = lifetimes[output_index];
        if (!output_lifetime.valid()) {
          continue;
        }
        if (output_lifetime.last_used_at > input_lifetime.last_used_at) {
          input_lifetime.last_used_at = output_lifetime.last_used_at;
          changed = true;
        }
      }
    }
  }

  for (size_t stage_idx = 0; stage_idx < stages.size(); ++stage_idx) {
    const auto &stage = stages[stage_idx];
    if (!stage.descriptor ||
        (!stage.descriptor->tensor_view_only &&
         !descriptor_outputs_share_first_input_storage(stage.descriptor))) {
      continue;
    }
    const bool has_internal_source =
        !stage.inputs.empty() &&
        stage.inputs.front().kind == FusedOutputLifetimeInputRef::Kind::Output &&
        stage.inputs.front().index < lifetimes.size() &&
        lifetimes[stage.inputs.front().index].valid();
    for (const auto output_index : stage.output_indices) {
      if (output_index >= lifetimes.size()) {
        continue;
      }
      auto &output_lifetime = lifetimes[output_index];
      if (!output_lifetime.valid()) {
        continue;
      }
      output_lifetime.requires_buffer = false;
      if (has_internal_source) {
        output_lifetime.storage_source_output = stage.inputs.front().index;
      }
    }
  }

  return std::any_of(lifetimes.begin(), lifetimes.end(),
                     [](const RuntimeOutputLifetime &lifetime) {
                       return lifetime.valid();
                     })
             ? lifetimes
             : std::vector<RuntimeOutputLifetime>{};
}

}  // namespace gfx_plugin
}  // namespace ov
