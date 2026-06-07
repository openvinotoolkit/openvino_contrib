// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/memory_plan.hpp"

#include <algorithm>
#include <limits>
#include <iomanip>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

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

std::string make_source_region_id(std::string_view prefix, const ov::Node &node,
                                  size_t output_idx) {
  std::ostringstream material;
  material << node.get_friendly_name() << "#" << node.get_type_name() << "#"
           << output_idx;
  std::ostringstream os;
  os << prefix << "_" << hex64(stable_hash64(material.str())) << ".output_"
     << output_idx;
  return os.str();
}

std::string make_logical_tensor_name(const PlannedOperation &op,
                                     std::string_view role, size_t index) {
  std::ostringstream os;
  os << op.node_name << "." << role << index;
  return os.str();
}

std::string make_source_logical_tensor_name(const ov::Node &node,
                                            size_t output_idx) {
  std::ostringstream os;
  os << node.get_friendly_name() << ".output" << output_idx;
  return os.str();
}

struct OutputPortKey {
  const ov::Node *node = nullptr;
  size_t output_idx = 0;

  bool operator==(const OutputPortKey &other) const noexcept {
    return node == other.node && output_idx == other.output_idx;
  }
};

struct OutputPortKeyHash {
  size_t operator()(const OutputPortKey &key) const noexcept {
    const auto pointer_hash =
        static_cast<size_t>(reinterpret_cast<std::uintptr_t>(key.node) >> 4);
    return pointer_hash ^ (key.output_idx + 0x9e3779b97f4a7c15ull +
                           (pointer_hash << 6) + (pointer_hash >> 2));
  }
};

struct PlannedOutputRef {
  size_t stage_id = 0;
  size_t output_idx = 0;
};

struct SourceRegionSummary {
  MemoryRegionKind kind = MemoryRegionKind::ExternalTensor;
  std::string region_id;
  std::string logical_tensor_name;
  std::string element_type;
  std::string partial_shape;
  std::string layout = "logical";
  size_t first_stage = std::numeric_limits<size_t>::max();
  size_t last_stage = 0;
};

class LoweringPlanMemoryGraph final {
public:
  explicit LoweringPlanMemoryGraph(const LoweringPlan &plan) : m_plan(plan) {
    index_planned_outputs();
    index_consumers();
  }

  std::string input_region_id(size_t stage_id, size_t input_idx) const {
    if (auto producer = planned_input_producer(stage_id, input_idx)) {
      return memory_region_id_for_stage_output(producer->stage_id,
                                               producer->output_idx);
    }
    if (const auto key = input_source_key(stage_id, input_idx); key.node) {
      return source_region_id(key);
    }
    return make_region_id(stage_id, "input", input_idx);
  }

  std::string output_region_id(size_t stage_id, size_t output_idx) const {
    return memory_region_id_for_stage_output(stage_id, output_idx);
  }

  std::vector<SourceRegionSummary> source_regions() const {
    std::unordered_map<OutputPortKey, SourceRegionSummary, OutputPortKeyHash>
        summaries;
    for (size_t stage_id = 0; stage_id < m_plan.operations.size(); ++stage_id) {
      const auto &op = m_plan.operations[stage_id];
      for (size_t input_idx = 0; input_idx < op.input_element_types.size();
           ++input_idx) {
        if (planned_input_producer(stage_id, input_idx)) {
          continue;
        }
        const auto key = input_source_key(stage_id, input_idx);
        if (!key.node) {
          SourceRegionSummary summary;
          summary.kind = MemoryRegionKind::ExternalTensor;
          summary.region_id = make_region_id(stage_id, "input", input_idx);
          summary.logical_tensor_name =
              make_logical_tensor_name(op, "input", input_idx);
          summary.element_type = op.input_element_types[input_idx];
          summary.partial_shape = input_idx < op.input_shapes.size()
                                      ? op.input_shapes[input_idx]
                                      : std::string{"?"};
          summary.layout =
              std::string(tensor_layout_kind_to_string(op.layout.kind));
          summary.first_stage = stage_id;
          summary.last_stage = stage_id;
          summaries[{nullptr, summaries.size()}] = std::move(summary);
          continue;
        }

        auto &summary = summaries[key];
        if (summary.region_id.empty()) {
          summary.kind = source_region_kind(*key.node);
          summary.region_id = source_region_id(key);
          summary.logical_tensor_name =
              make_source_logical_tensor_name(*key.node, key.output_idx);
          summary.element_type = op.input_element_types[input_idx];
          summary.partial_shape = input_idx < op.input_shapes.size()
                                      ? op.input_shapes[input_idx]
                                      : std::string{"?"};
          summary.layout =
              std::string(tensor_layout_kind_to_string(op.layout.kind));
        }
        summary.first_stage = std::min(summary.first_stage, stage_id);
        summary.last_stage = std::max(summary.last_stage, stage_id);
      }
    }

    std::vector<SourceRegionSummary> result;
    result.reserve(summaries.size());
    for (auto &entry : summaries) {
      if (entry.second.first_stage == std::numeric_limits<size_t>::max()) {
        entry.second.first_stage = 0;
      }
      result.push_back(std::move(entry.second));
    }
    std::sort(result.begin(), result.end(),
              [](const SourceRegionSummary &lhs,
                 const SourceRegionSummary &rhs) {
                return lhs.region_id < rhs.region_id;
              });
    return result;
  }

  LifetimeInterval planned_output_lifetime(size_t stage_id,
                                           size_t output_idx) const {
    const size_t final_stage =
        m_plan.operations.empty() ? 0 : m_plan.operations.size() - 1;
    size_t last_stage = stage_id;
    const auto key = planned_output_key(stage_id, output_idx);
    if (auto it = m_consumer_stages.find(key); it != m_consumer_stages.end()) {
      for (const auto consumer_stage : it->second) {
        last_stage = std::max(last_stage, consumer_stage);
      }
    }
    if (planned_output_escapes_to_model_boundary(stage_id, output_idx)) {
      last_stage = std::max(last_stage, final_stage);
    }
    return {output_region_kind(stage_id) == MemoryRegionKind::TransientTensor
                ? stage_id
                : 0,
            last_stage};
  }

  MemoryRegionKind output_region_kind(size_t stage_id) const {
    if (stage_id >= m_plan.operations.size()) {
      return MemoryRegionKind::TransientTensor;
    }
    const auto &op = m_plan.operations[stage_id];
    if (ov::as_type_ptr<const ov::op::v0::Parameter>(op.source_node)) {
      return MemoryRegionKind::ExternalTensor;
    }
    if (ov::as_type_ptr<const ov::op::v0::Constant>(op.source_node)) {
      return MemoryRegionKind::ImmutableTensor;
    }
    return MemoryRegionKind::TransientTensor;
  }

private:
  OutputPortKey planned_output_key(size_t stage_id, size_t output_idx) const {
    if (stage_id >= m_plan.operations.size()) {
      return {};
    }
    const auto &op = m_plan.operations[stage_id];
    if (!op.source_node || output_idx >= op.source_node->get_output_size()) {
      return {};
    }
    return {op.source_node.get(), output_idx};
  }

  OutputPortKey input_source_key(size_t stage_id, size_t input_idx) const {
    if (stage_id >= m_plan.operations.size()) {
      return {};
    }
    const auto &op = m_plan.operations[stage_id];
    if (!op.source_node || input_idx >= op.source_node->get_input_size()) {
      return {};
    }
    const auto source = op.source_node->input_value(input_idx);
    return {source.get_node(), source.get_index()};
  }

  std::optional<PlannedOutputRef> planned_input_producer(
      size_t stage_id, size_t input_idx) const {
    const auto key = input_source_key(stage_id, input_idx);
    if (!key.node) {
      return std::nullopt;
    }
    const auto it = m_planned_outputs.find(key);
    if (it == m_planned_outputs.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  std::string source_region_id(const OutputPortKey &key) const {
    if (!key.node) {
      return {};
    }
    const auto kind = source_region_kind(*key.node);
    const auto prefix =
        kind == MemoryRegionKind::ImmutableTensor ? "model_const" : "external";
    return make_source_region_id(prefix, *key.node, key.output_idx);
  }

  MemoryRegionKind source_region_kind(const ov::Node &node) const {
    if (dynamic_cast<const ov::op::v0::Constant *>(&node)) {
      return MemoryRegionKind::ImmutableTensor;
    }
    return MemoryRegionKind::ExternalTensor;
  }

  void index_planned_outputs() {
    for (size_t stage_id = 0; stage_id < m_plan.operations.size(); ++stage_id) {
      const auto &op = m_plan.operations[stage_id];
      if (!op.source_node) {
        continue;
      }
      for (size_t output_idx = 0; output_idx < op.source_node->get_output_size();
           ++output_idx) {
        m_planned_outputs[{op.source_node.get(), output_idx}] =
            PlannedOutputRef{stage_id, output_idx};
      }
    }
  }

  void index_consumers() {
    for (size_t stage_id = 0; stage_id < m_plan.operations.size(); ++stage_id) {
      const auto &op = m_plan.operations[stage_id];
      if (!op.source_node) {
        continue;
      }
      for (size_t input_idx = 0; input_idx < op.source_node->get_input_size();
           ++input_idx) {
        const auto key = input_source_key(stage_id, input_idx);
        if (m_planned_outputs.find(key) != m_planned_outputs.end()) {
          m_consumer_stages[key].push_back(stage_id);
        }
      }
    }
  }

  bool planned_output_escapes_to_model_boundary(size_t stage_id,
                                                size_t output_idx) const {
    if (stage_id >= m_plan.operations.size()) {
      return false;
    }
    const auto &op = m_plan.operations[stage_id];
    if (!op.source_node || output_idx >= op.source_node->get_output_size()) {
      return false;
    }
    for (const auto &target :
         op.source_node->output(output_idx).get_target_inputs()) {
      const auto *consumer = target.get_node();
      if (!consumer) {
        continue;
      }
      if (dynamic_cast<const ov::op::v0::Result *>(consumer)) {
        return true;
      }
      const auto consumer_it =
          std::find_if(m_plan.operations.begin(), m_plan.operations.end(),
                       [&](const PlannedOperation &planned_op) {
                         return planned_op.source_node.get() == consumer;
                       });
      if (consumer_it == m_plan.operations.end()) {
        return true;
      }
    }
    return false;
  }

  const LoweringPlan &m_plan;
  std::unordered_map<OutputPortKey, PlannedOutputRef, OutputPortKeyHash>
      m_planned_outputs;
  std::unordered_map<OutputPortKey, std::vector<size_t>, OutputPortKeyHash>
      m_consumer_stages;
};

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

const MemoryRegion *find_region_by_id(const std::vector<MemoryRegion> &regions,
                                      const std::string &region_id) {
  const auto it =
      std::find_if(regions.begin(), regions.end(),
                   [&](const MemoryRegion &region) {
                     return region.region_id == region_id;
                   });
  return it == regions.end() ? nullptr : &*it;
}

bool alias_regions_compatible(const MemoryRegion &lhs,
                              const MemoryRegion &rhs) {
  return lhs.kind == MemoryRegionKind::TransientTensor &&
         rhs.kind == MemoryRegionKind::TransientTensor &&
         lhs.storage_kind == rhs.storage_kind &&
         lhs.element_type == rhs.element_type && lhs.layout == rhs.layout &&
         lhs.partial_shape == rhs.partial_shape &&
         !lhs.lifetime.overlaps(rhs.lifetime);
}

void assign_transient_alias_groups(MemoryPlan &plan) {
  size_t next_group_id = 0;
  for (auto &region : plan.regions) {
    if (region.kind != MemoryRegionKind::TransientTensor) {
      region.alias_group = region.region_id;
      attach_alias_group(plan, region.alias_group, region.region_id,
                         /*output_aliasing=*/false);
      continue;
    }

    AliasGroup *compatible_group = nullptr;
    for (auto &group : plan.alias_groups) {
      bool compatible = true;
      for (const auto &region_id : group.region_ids) {
        const auto *other = find_region_by_id(plan.regions, region_id);
        if (!other || !alias_regions_compatible(region, *other)) {
          compatible = false;
          break;
        }
      }
      if (compatible) {
        compatible_group = &group;
        break;
      }
    }

    if (!compatible_group) {
      AliasGroup group;
      group.group_id = "transient_alias_" + std::to_string(next_group_id++);
      plan.alias_groups.push_back(std::move(group));
      compatible_group = &plan.alias_groups.back();
    }
    region.alias_group = compatible_group->group_id;
    add_region_id_once(compatible_group->region_ids, region.region_id);
  }
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
    if (!group.output_aliasing) {
      for (size_t i = 0; i < group.region_ids.size(); ++i) {
        const auto *lhs = find_region_by_id(regions, group.region_ids[i]);
        if (!lhs || lhs->kind != MemoryRegionKind::TransientTensor) {
          continue;
        }
        for (size_t j = i + 1; j < group.region_ids.size(); ++j) {
          const auto *rhs = find_region_by_id(regions, group.region_ids[j]);
          if (!rhs || rhs->kind != MemoryRegionKind::TransientTensor) {
            continue;
          }
          if (lhs->lifetime.overlaps(rhs->lifetime)) {
            result.diagnostics.push_back(
                "alias group " + group.group_id +
                " contains overlapping transient lifetimes");
          }
        }
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

const MemoryRegion *MemoryPlan::find_region(std::string_view region_id) const {
  const auto it =
      std::find_if(regions.begin(), regions.end(),
                   [&](const MemoryRegion &region) {
                     return region.region_id == region_id;
                   });
  return it == regions.end() ? nullptr : &*it;
}

bool MemoryPlan::has_region(std::string_view region_id) const {
  return find_region(region_id) != nullptr;
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
  const LoweringPlanMemoryGraph graph(plan);

  TransientArena transient_arena;
  transient_arena.arena_id = "transient_device_buffer_arena";
  transient_arena.storage_kind = "device_buffer";

  for (const auto &source_region : graph.source_regions()) {
    MemoryRegion region;
    region.region_id = source_region.region_id;
    region.logical_tensor_name = source_region.logical_tensor_name;
    region.kind = source_region.kind;
    region.element_type = source_region.element_type;
    region.partial_shape = source_region.partial_shape;
    region.layout = source_region.layout;
    region.storage_kind = "device_buffer";
    region.alias_group = region.region_id;
    region.lifetime = {source_region.first_stage, source_region.last_stage};
    region.external_binding = region.kind == MemoryRegionKind::ExternalTensor;
    attach_alias_group(memory_plan, region.alias_group, region.region_id,
                       /*output_aliasing=*/false);
    memory_plan.regions.push_back(std::move(region));
  }

  for (size_t stage_id = 0; stage_id < plan.operations.size(); ++stage_id) {
    const auto &op = plan.operations[stage_id];
    for (size_t input_idx = 0; input_idx < op.input_element_types.size();
         ++input_idx) {
      const auto region_id = graph.input_region_id(stage_id, input_idx);
      if (!region_id.empty() && memory_plan.has_region(region_id)) {
        continue;
      }
      MemoryRegion region;
      region.region_id =
          region_id.empty() ? make_region_id(stage_id, "input", input_idx)
                            : region_id;
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
      region.lifetime = {stage_id, stage_id};
      region.external_binding = true;
      attach_alias_group(memory_plan, region.alias_group, region.region_id,
                         /*output_aliasing=*/false);
      memory_plan.regions.push_back(std::move(region));
    }

    for (size_t output_idx = 0; output_idx < op.output_element_types.size();
         ++output_idx) {
      MemoryRegion region;
      region.region_id = graph.output_region_id(stage_id, output_idx);
      region.logical_tensor_name =
          make_logical_tensor_name(op, "output", output_idx);
      region.kind = graph.output_region_kind(stage_id);
      region.element_type = op.output_element_types[output_idx];
      region.partial_shape = output_idx < op.output_shapes.size()
                                  ? op.output_shapes[output_idx]
                                  : std::string{"?"};
      region.layout = std::string(tensor_layout_kind_to_string(op.layout.kind));
      region.storage_kind = "device_buffer";
      region.lifetime = graph.planned_output_lifetime(stage_id, output_idx);
      region.external_binding = region.kind == MemoryRegionKind::ExternalTensor;
      memory_plan.regions.push_back(std::move(region));
    }
  }

  assign_transient_alias_groups(memory_plan);

  for (const auto &region : memory_plan.regions) {
    if (region.kind == MemoryRegionKind::TransientTensor) {
      transient_arena.region_ids.push_back(region.region_id);
    }
  }

  if (!transient_arena.region_ids.empty()) {
    memory_plan.transient_arenas.push_back(std::move(transient_arena));
  }
  return memory_plan;
}

std::string memory_region_id_for_stage_input(const LoweringPlan &plan,
                                             size_t stage_id,
                                             size_t input_idx) {
  return LoweringPlanMemoryGraph(plan).input_region_id(stage_id, input_idx);
}

std::string memory_region_id_for_stage_output(size_t stage_id,
                                              size_t output_idx) {
  return make_region_id(stage_id, "output", output_idx);
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
