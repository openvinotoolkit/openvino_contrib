// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/kernel_registry.hpp"

#include <utility>

namespace ov {
namespace gfx_plugin {
namespace compiler {

namespace {

KernelUnit make_metal_lowering_unit(const BackendTarget& target) {
    return KernelUnit::describe(LoweringRouteKind::BackendLowering,
                                KernelUnitKind::BackendLowering,
                                "metal_lowering",
                                target.backend_id(),
                                "apple_mps_mpsgraph_msl_transition");
}

}  // namespace

KernelRegistry make_metal_kernel_registry(const BackendTarget& target) {
    auto units = make_common_kernel_units(target);
    units.push_back(make_metal_lowering_unit(target));
    return KernelRegistry(target, std::move(units));
}

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
