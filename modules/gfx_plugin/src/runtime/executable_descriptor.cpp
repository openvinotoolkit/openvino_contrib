// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/executable_descriptor.hpp"

#include <algorithm>
#include <string_view>

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool runtime_descriptor_stateful_stage_kind(
    std::string_view stateful_effect) noexcept {
  return stateful_effect == "assign" || stateful_effect == "read_value";
}

bool runtime_descriptor_view_stage_contract(
    const RuntimeStageExecutableDescriptor &descriptor) noexcept {
  return descriptor.origin == KernelArtifactOrigin::Metadata &&
         descriptor.payload_kind == KernelArtifactPayloadKind::None &&
         descriptor.kernel_id == "metadata" && descriptor.tensor_view_only;
}

} // namespace

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

std::vector<GfxKernelBufferRole> materialize_descriptor_launch_roles(
    const KernelLaunchPlanDescriptor &plan, std::string_view stage_name) {
  OPENVINO_ASSERT(plan.valid && !plan.buffer_roles.empty(),
                  "GFX: source descriptor launch plan is missing for ",
                  stage_name);
  std::vector<GfxKernelBufferRole> roles;
  roles.reserve(plan.buffer_roles.size());
  for (const auto &role_name : plan.buffer_roles) {
    const auto role = kernel_buffer_role_from_descriptor_name(role_name);
    OPENVINO_ASSERT(role != GfxKernelBufferRole::Unknown,
                    "GFX: source descriptor launch plan has unknown role ",
                    role_name, " for ", stage_name);
    roles.push_back(role);
  }
  return roles;
}

bool runtime_descriptor_source_payload_kind(
    KernelArtifactPayloadKind kind) noexcept {
  return kind == KernelArtifactPayloadKind::MslSource ||
         kind == KernelArtifactPayloadKind::OpenClSource;
}

bool runtime_descriptor_payload_kind_requires_payload(
    KernelArtifactPayloadKind kind) noexcept {
  return kind == KernelArtifactPayloadKind::VendorDescriptor ||
         runtime_descriptor_source_payload_kind(kind);
}

bool runtime_stage_descriptor_is_materializable(
    const RuntimeStageExecutableDescriptor &descriptor) noexcept {
  if (runtime_descriptor_stateful_stage_kind(descriptor.stateful_effect)) {
    return true;
  }
  if (runtime_descriptor_view_stage_contract(descriptor)) {
    return true;
  }
  if (!runtime_descriptor_payload_kind_requires_payload(
          descriptor.payload_kind)) {
    return false;
  }
  if (!descriptor.payload || !descriptor.payload->valid()) {
    return false;
  }
  if (descriptor.payload_kind == KernelArtifactPayloadKind::VendorDescriptor) {
    return descriptor.origin == KernelArtifactOrigin::VendorPrimitive;
  }
  return descriptor.origin == KernelArtifactOrigin::Generated ||
         descriptor.origin == KernelArtifactOrigin::HandwrittenException;
}

} // namespace gfx_plugin
} // namespace ov
