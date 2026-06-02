// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "compiler/lowering_planner.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

enum class MemoryRegionKind {
  ExternalTensor,
  TransientTensor,
  ImmutableTensor,
};

struct LifetimeInterval {
  size_t first_stage = 0;
  size_t last_stage = 0;

  bool valid() const noexcept { return first_stage <= last_stage; }
  bool overlaps(const LifetimeInterval &other) const noexcept {
    return first_stage <= other.last_stage && other.first_stage <= last_stage;
  }
};

struct MemoryRegion {
  std::string region_id;
  std::string logical_tensor_name;
  MemoryRegionKind kind = MemoryRegionKind::TransientTensor;
  std::string element_type;
  std::string partial_shape;
  std::string layout = "logical";
  std::string storage_kind = "device_buffer";
  std::string alias_group;
  LifetimeInterval lifetime;
  bool external_binding = false;
  bool host_visible = false;
};

struct AliasGroup {
  std::string group_id;
  std::vector<std::string> region_ids;
  bool output_aliasing = false;
};

struct TransientArena {
  std::string arena_id;
  std::string storage_kind = "device_buffer";
  std::vector<std::string> region_ids;
};

struct MemoryPlanVerificationResult {
  std::vector<std::string> diagnostics;

  bool valid() const noexcept { return diagnostics.empty(); }
};

struct MemoryPlan {
  uint32_t schema_version = 1;
  std::vector<MemoryRegion> regions;
  std::vector<AliasGroup> alias_groups;
  std::vector<TransientArena> transient_arenas;
  bool hidden_host_copies_allowed = false;

  MemoryPlanVerificationResult verify() const;
  bool valid() const;
  bool has_region(std::string_view region_id) const;
  bool has_alias_group(std::string_view group_id) const;
};

class MemoryPlanBuilder final {
public:
  MemoryPlan build(const LoweringPlan &plan) const;
};

std::string_view memory_region_kind_to_string(MemoryRegionKind kind) noexcept;
std::string make_memory_plan_fingerprint(const MemoryPlan &plan);

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
