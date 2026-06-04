// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/executable_descriptor.hpp"

#include <algorithm>
#include <string_view>

namespace ov {
namespace gfx_plugin {

bool RuntimeMemoryPlanDescriptor::has_region(std::string_view region_id) const {
  return std::any_of(regions.begin(), regions.end(),
                     [&](const RuntimeMemoryRegionDescriptor &region) {
                       return region.region_id == region_id;
                     });
}

bool RuntimeMemoryPlanDescriptor::has_alias_group(
    std::string_view group_id) const {
  return std::any_of(alias_groups.begin(), alias_groups.end(),
                     [&](const RuntimeMemoryAliasGroupDescriptor &group) {
                       return group.group_id == group_id;
                     });
}

} // namespace gfx_plugin
} // namespace ov
