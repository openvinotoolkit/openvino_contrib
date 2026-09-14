// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_stage_placement.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

class OpenClStagePlacementPolicy final : public StagePlacementPolicy {
public:
  GfxStagePlacementPlan
  select_placement(const StagePlacementQuery &query) const override {
    return make_stage_placement(GfxStageBackendDomain::OpenCl,
                                GfxStageStorageKind::Buffer, query.stage_type,
                                /*vendor_primitive=*/false,
                                /*custom_kernel=*/true);
  }
};

} // namespace

std::shared_ptr<const StagePlacementPolicy>
make_opencl_stage_placement_policy() {
  static const auto policy = std::make_shared<OpenClStagePlacementPolicy>();
  return policy;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
