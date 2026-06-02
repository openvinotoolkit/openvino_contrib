// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/memory_plan.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

constexpr uint32_t kMemoryPlanSchemaVersion = 1;

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

std::string hex64(uint64_t value) {
  std::ostringstream os;
  os << std::hex << std::setw(16) << std::setfill('0') << value;
  return os.str();
}

std::string make_region_id(size_t stage_id, std::string_view role, size_t index) {
  std::ostringstream os;
  os << "stage_" << stage_id << "." << role << "_" << index;
  return os.str();
}

std::string make_logical_tensor_name(const PlannedOperation &op,
                                     std::string_view role, size_t index) {
  std::ostringstream os;
  os << op.node_name << "." << role << index;
  return os.str();
}

void add_region_id_once(std::vector<std::string> &ids, const std::string &id) {
  if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
    ids.push_back(id);
  }
}

AliasGroup *find_alias_group(MemoryPlan &plan, std::string_view group_id) {
  for (auto &group : plan.alias_groups) {
    if (group.group_id == group_id) {
      return &group;
    }
  }
  return nullptr;
}

void attach_alias_group(MemoryPlan &plan, std::string group_id,
                        const std::string &region_id, bool output_aliasing) {
  auto *group = find_alias_group(plan, group_id);
  if (!group) {
    AliasGroup new_group;
    new_group.group_id = std::move(group_id);
    new_group.output_aliasing = output_aliasing;
    plan.alias_groups.push_back(std::move(new_group));
    group = &plan.alias_groups.back();
  }
  group->output_aliasing = group->output_aliasing || output_aliasing;
  add_region_id_once(group->region_ids, region_id);
}

void require_nonempty(MemoryPlanVerificationResult &result,
                      std::string_view value, std::string diagnostic) {
  if (value.empty()) {
    result.diagnostics.push_back(std::move(diagnostic));
  }
}

bool region_id_exists(const std::unordered_set<std::string> &ids,
                      const std::string &id) {
  return ids.find(id) != ids.end();
}

} // namespace

std::string_view memory_region_kind_to_string(MemoryRegionKind kind) noexcept {
  switch (kind) {
  case MemoryRegionKind::ExternalTensor:
    return "external_tensor";
  case MemoryRegionKind::TransientTensor:
    return "transient_tensor";
  case MemoryRegionKind::ImmutableTensor:
    return "immutable_tensor";
  }
  return "transient_tensor";
}

MemoryPlanVerificationResult MemoryPlan::verify() const {
  MemoryPlanVerificationResult result;
  if (schema_version != kMemoryPlanSchemaVersion) {
    result.diagnostics.emplace_back("memory plan schema version mismatch");
  }
  if (hidden_host_copies_allowed) {
    result.diagnostics.emplace_back("memory plan allows hidden host copies");
  }

  std::unordered_set<std::string> region_ids;
  region_ids.reserve(regions.size());
  for (const auto &region : regions) {
    require_nonempty(result, region.region_id,
                     "memory region has empty region id");
    require_nonempty(result, region.logical_tensor_name,
                     "memory region has empty logical tensor name");
    require_nonempty(result, region.element_type,
                     "memory region has empty element type");
    require_nonempty(result, region.partial_shape,
                     "memory region has empty partial shape");
    require_nonempty(result, region.layout, "memory region has empty layout");
    require_nonempty(result, region.storage_kind,
                     "memory region has empty storage kind");
    require_nonempty(result, region.alias_group,
                     "memory region has empty alias group");
    if (!region.lifetime.valid()) {
      result.diagnostics.push_back("memory region " + region.region_id +
                                   " has invalid lifetime interval");
    }
    if (!region.region_id.empty() && !region_ids.insert(region.region_id).second) {
      result.diagnostics.push_back("duplicate memory region id " +
                                   region.region_id);
    }
  }

  std::unordered_set<std::string> alias_ids;
  alias_ids.reserve(alias_groups.size());
  for (const auto &group : alias_groups) {
    require_nonempty(result, group.group_id, "alias group has empty id");
    if (!group.group_id.empty() && !alias_ids.insert(group.group_id).second) {
      result.diagnostics.push_back("duplicate alias group id " + group.group_id);
    }
    if (group.region_ids.empty()) {
      result.diagnostics.push_back("alias group " + group.group_id +
                                   " has no regions");
    }
    for (const auto &region_id : group.region_ids) {
      if (!region_id_exists(region_ids, region_id)) {
        result.diagnostics.push_back("alias group " + group.group_id +
                                     " references unknown region " + region_id);
      }
    }
  }

  std::unordered_set<std::string> arena_ids;
  arena_ids.reserve(transient_arenas.size());
  for (const auto &arena : transient_arenas) {
    require_nonempty(result, arena.arena_id, "transient arena has empty id");
    require_nonempty(result, arena.storage_kind,
                     "transient arena has empty storage kind");
    if (!arena.arena_id.empty() && !arena_ids.insert(arena.arena_id).second) {
      result.diagnostics.push_back("duplicate transient arena id " +
                                   arena.arena_id);
    }
    for (const auto &region_id : arena.region_ids) {
      if (!region_id_exists(region_ids, region_id)) {
        result.diagnostics.push_back("transient arena " + arena.arena_id +
                                     " references unknown region " + region_id);
        continue;
      }
      const auto region_it =
          std::find_if(regions.begin(), regions.end(),
                       [&](const MemoryRegion &region) {
                         return region.region_id == region_id;
                       });
      if (region_it != regions.end() &&
          region_it->kind != MemoryRegionKind::TransientTensor) {
        result.diagnostics.push_back("transient arena " + arena.arena_id +
                                     " references non-transient region " +
                                     region_id);
      }
    }
  }
  return result;
}

bool MemoryPlan::valid() const { return verify().valid(); }

bool MemoryPlan::has_region(std::string_view region_id) const {
  return std::any_of(regions.begin(), regions.end(),
                     [&](const MemoryRegion &region) {
                       return region.region_id == region_id;
                     });
}

bool MemoryPlan::has_alias_group(std::string_view group_id) const {
  return std::any_of(alias_groups.begin(), alias_groups.end(),
                     [&](const AliasGroup &group) {
                       return group.group_id == group_id;
                     });
}

MemoryPlan MemoryPlanBuilder::build(const LoweringPlan &plan) const {
  MemoryPlan memory_plan;
  memory_plan.schema_version = kMemoryPlanSchemaVersion;

  TransientArena transient_arena;
  transient_arena.arena_id = "transient_device_buffer_arena";
  transient_arena.storage_kind = "device_buffer";

  const size_t final_stage =
      plan.operations.empty() ? 0 : plan.operations.size() - 1;
  for (size_t stage_id = 0; stage_id < plan.operations.size(); ++stage_id) {
    const auto &op = plan.operations[stage_id];
    for (size_t input_idx = 0; input_idx < op.input_element_types.size();
         ++input_idx) {
      MemoryRegion region;
      region.region_id = make_region_id(stage_id, "input", input_idx);
      region.logical_tensor_name =
          make_logical_tensor_name(op, "input", input_idx);
      region.kind = MemoryRegionKind::ExternalTensor;
      region.element_type = op.input_element_types[input_idx];
      region.partial_shape = input_idx < op.input_shapes.size()
                                 ? op.input_shapes[input_idx]
                                 : std::string{"?"};
      region.layout = std::string(tensor_layout_kind_to_string(op.layout.kind));
      region.storage_kind = "device_buffer";
      region.alias_group = region.region_id;
      region.lifetime = {0, stage_id};
      region.external_binding = true;
      attach_alias_group(memory_plan, region.alias_group, region.region_id,
                         /*output_aliasing=*/false);
      memory_plan.regions.push_back(std::move(region));
    }

    const std::string output_alias_group = "stage_" + std::to_string(stage_id);
    for (size_t output_idx = 0; output_idx < op.output_element_types.size();
         ++output_idx) {
      MemoryRegion region;
      region.region_id = make_region_id(stage_id, "output", output_idx);
      region.logical_tensor_name =
          make_logical_tensor_name(op, "output", output_idx);
      region.kind = MemoryRegionKind::TransientTensor;
      region.element_type = op.output_element_types[output_idx];
      region.partial_shape = output_idx < op.output_shapes.size()
                                  ? op.output_shapes[output_idx]
                                  : std::string{"?"};
      region.layout = std::string(tensor_layout_kind_to_string(op.layout.kind));
      region.storage_kind = "device_buffer";
      region.alias_group = output_alias_group;
      region.lifetime = {stage_id, final_stage};
      attach_alias_group(memory_plan, output_alias_group, region.region_id,
                         /*output_aliasing=*/false);
      transient_arena.region_ids.push_back(region.region_id);
      memory_plan.regions.push_back(std::move(region));
    }
  }

  if (!transient_arena.region_ids.empty()) {
    memory_plan.transient_arenas.push_back(std::move(transient_arena));
  }
  return memory_plan;
}

std::string make_memory_plan_fingerprint(const MemoryPlan &plan) {
  std::ostringstream material;
  append_field(material, std::to_string(plan.schema_version));
  append_bool(material, plan.hidden_host_copies_allowed);
  append_field(material, std::to_string(plan.regions.size()));
  for (const auto &region : plan.regions) {
    append_field(material, region.region_id);
    append_field(material, region.logical_tensor_name);
    append_field(material, memory_region_kind_to_string(region.kind));
    append_field(material, region.element_type);
    append_field(material, region.partial_shape);
    append_field(material, region.layout);
    append_field(material, region.storage_kind);
    append_field(material, region.alias_group);
    append_field(material, std::to_string(region.lifetime.first_stage));
    append_field(material, std::to_string(region.lifetime.last_stage));
    append_bool(material, region.external_binding);
    append_bool(material, region.host_visible);
  }
  append_field(material, std::to_string(plan.alias_groups.size()));
  for (const auto &group : plan.alias_groups) {
    append_field(material, group.group_id);
    append_bool(material, group.output_aliasing);
    for (const auto &region_id : group.region_ids) {
      append_field(material, region_id);
    }
  }
  append_field(material, std::to_string(plan.transient_arenas.size()));
  for (const auto &arena : plan.transient_arenas) {
    append_field(material, arena.arena_id);
    append_field(material, arena.storage_kind);
    for (const auto &region_id : arena.region_ids) {
      append_field(material, region_id);
    }
  }
  return hex64(stable_hash64(material.str()));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
