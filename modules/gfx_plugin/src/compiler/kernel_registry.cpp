// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/kernel_registry.hpp"

#include <utility>

namespace ov {
namespace gfx_plugin {
namespace compiler {

namespace {

KernelUnit make_common_unit(const BackendTarget &target) {
  return KernelUnit::describe(LoweringRouteKind::Common, KernelUnitKind::Common,
                              "common", target.backend_id(), "common_io");
}

KernelUnit make_metadata_unit(const BackendTarget &target) {
  return KernelUnit::describe(LoweringRouteKind::Metadata,
                              KernelUnitKind::Metadata, "metadata",
                              target.backend_id(), "metadata");
}

bool same_unit_key(const KernelUnit &lhs, const KernelUnit &rhs) {
  return lhs.id() == rhs.id() && lhs.backend_domain() == rhs.backend_domain();
}

} // namespace

KernelRegistry::KernelRegistry(BackendTarget target,
                               std::vector<KernelUnit> units)
    : m_target(std::move(target)), m_units(std::move(units)) {}

std::vector<KernelUnit> make_common_kernel_units(const BackendTarget &target) {
  std::vector<KernelUnit> units;
  units.push_back(make_common_unit(target));
  units.push_back(make_metadata_unit(target));
  return units;
}

KernelRegistry make_common_kernel_registry(const BackendTarget &target) {
  return KernelRegistry(target, make_common_kernel_units(target));
}

KernelUnit KernelRegistry::resolve(LoweringRouteKind route_kind,
                                   std::string_view unit_id) const {
  KernelUnit route_match;
  size_t route_match_count = 0;
  for (const auto &unit : m_units) {
    if (unit.route_kind() != route_kind) {
      continue;
    }
    if (!unit_id.empty() && unit.id() == unit_id) {
      return unit;
    }
    if (unit_id.empty()) {
      route_match = unit;
      ++route_match_count;
    }
  }
  if (route_match_count == 1) {
    return route_match;
  }
  return {};
}

KernelRegistryAudit KernelRegistry::audit() const {
  KernelRegistryAudit audit;
  for (size_t i = 0; i < m_units.size(); ++i) {
    const auto &unit = m_units[i];
    if (!unit.valid() || unit.backend_domain().empty() ||
        unit.op_family().empty()) {
      audit.diagnostics.push_back("incomplete kernel unit: " + unit.id());
    }
    if (unit.kind() == KernelUnitKind::HandwrittenException) {
      ++audit.handwritten_exception_count;
      if (unit.route_kind() != LoweringRouteKind::HandwrittenKernelException) {
        audit.diagnostics.push_back(
            "handwritten kernel unit has non-exception route: " + unit.id());
      }
      if (!unit.exception_contract().valid()) {
        audit.diagnostics.push_back(
            "handwritten kernel unit is missing exception contract: " +
            unit.id());
      }
    } else if (unit.route_kind() ==
               LoweringRouteKind::HandwrittenKernelException) {
      audit.diagnostics.push_back(
          "handwritten exception route has non-exception unit kind: " +
          unit.id());
    }
    for (size_t j = i + 1; j < m_units.size(); ++j) {
      if (same_unit_key(unit, m_units[j])) {
        audit.diagnostics.push_back("duplicate kernel unit: " + unit.id());
      }
    }
  }
  return audit;
}

size_t KernelRegistry::route_count(LoweringRouteKind route_kind) const {
  size_t count = 0;
  for (const auto &unit : m_units) {
    if (unit.route_kind() == route_kind) {
      ++count;
    }
  }
  return count;
}

size_t KernelRegistry::handwritten_exception_count() const {
  return audit().handwritten_exception_count;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
