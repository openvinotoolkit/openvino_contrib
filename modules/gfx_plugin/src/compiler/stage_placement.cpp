// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/stage_placement.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

const char *stage_backend_domain_key(GfxStageBackendDomain domain) {
  switch (domain) {
  case GfxStageBackendDomain::AppleMps:
    return "apple_mps";
  case GfxStageBackendDomain::AppleMsl:
    return "apple_msl";
  case GfxStageBackendDomain::OpenCl:
    return "opencl";
  case GfxStageBackendDomain::Unknown:
  default:
    return "unknown";
  }
}

const char *stage_storage_key(GfxStageStorageKind storage) {
  switch (storage) {
  case GfxStageStorageKind::Buffer:
    return "buffer";
  case GfxStageStorageKind::Image:
    return "image";
  case GfxStageStorageKind::Matrix:
    return "matrix";
  case GfxStageStorageKind::NDArray:
    return "ndarray";
  case GfxStageStorageKind::Alias:
    return "alias";
  case GfxStageStorageKind::Unknown:
  default:
    return "unknown";
  }
}

} // namespace

std::string make_stage_placement_key(GfxStageBackendDomain domain,
                                     GfxStageStorageKind storage,
                                     std::string_view stage_type) {
  std::string key(stage_backend_domain_key(domain));
  key += ":";
  key += stage_storage_key(storage);
  key += ":";
  key += stage_type;
  return key;
}

GfxStagePlacementPlan make_stage_placement(GfxStageBackendDomain domain,
                                           GfxStageStorageKind storage,
                                           std::string_view stage_type,
                                           bool vendor_primitive,
                                           bool custom_kernel) {
  GfxStagePlacementPlan plan{};
  plan.domain = domain;
  plan.storage = storage;
  plan.uses_vendor_primitive = vendor_primitive;
  plan.uses_custom_kernel = custom_kernel;
  plan.specialization_key = make_stage_placement_key(domain, storage, stage_type);
  return plan;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
