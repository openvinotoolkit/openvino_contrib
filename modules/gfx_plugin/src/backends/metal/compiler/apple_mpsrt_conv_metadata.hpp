// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "backends/metal/compiler/apple_stage_pipeline.hpp"
#include "backends/metal/compiler/apple_vendor_descriptors.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "openvino/core/node.hpp"
#include "backends/metal/common/mpsrt/gfx_mpsrt_abi.hpp"
#include "common/gfx_bias.hpp"
#include "compiler/stage_policy.hpp"

namespace ov {
namespace gfx_plugin {

enum class GfxConvMpsrtLoweringKind {
    None,
    MpsConv2D,
    MpsGroupConv2D,
};

inline uint32_t gfx_mpsrt_conv_fused_activation_code(ActivationKind kind) {
    return gfx_apple_mps_conv_fused_activation_code(kind);
}

inline bool gfx_mpsrt_conv_supports_fused_activation(ActivationKind kind) {
    return gfx_apple_mps_conv_supports_fused_activation(kind);
}

inline std::string gfx_mpsrt_canonical_conv_stage_type(const std::shared_ptr<const ov::Node>& node,
                                                       std::string_view fallback_stage_type) {
    return gfx_apple_mps_canonical_conv_stage_type(node, fallback_stage_type);
}

inline GfxConvMpsrtLoweringKind annotate_module_with_conv_mpsrt_plan(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const std::shared_ptr<const ov::Node>& node,
    std::string_view fallback_stage_type,
    bool has_bias = false,
    const BiasParams* bias_params = nullptr,
    bool has_activation = false,
    ActivationKind activation = ActivationKind::Identity) {
    if (!module ||
        plan.placement.domain != GfxStageBackendDomain::AppleMps ||
        !plan.placement.uses_vendor_primitive ||
        plan.placement.uses_custom_kernel) {
        return GfxConvMpsrtLoweringKind::None;
    }
    if (has_activation && !gfx_mpsrt_conv_supports_fused_activation(activation)) {
        return GfxConvMpsrtLoweringKind::None;
    }

    const auto stage_type = gfx_mpsrt_canonical_conv_stage_type(node, fallback_stage_type);
    GfxAppleMpsVendorPrimitiveContract contract{};
    if (!gfx_apple_make_mps_conv2d_contract(node, has_bias, bias_params, has_activation, activation, contract)) {
        return GfxConvMpsrtLoweringKind::None;
    }

    const auto materialized =
        materialize_apple_mps_vendor_contract_program(module, plan, stage_type, contract);
    if (!materialized.valid || !materialized.typed_program_materialized) {
        return GfxConvMpsrtLoweringKind::None;
    }
    GfxMpsrtModuleStagePlan stage_plan{};
    if (!read_module_mpsrt_stage_plan(module, stage_plan)) {
        return GfxConvMpsrtLoweringKind::None;
    }
    return stage_plan.stage.kind == GfxMpsrtStageKind::MPSConv2D
               ? GfxConvMpsrtLoweringKind::MpsConv2D
               : GfxConvMpsrtLoweringKind::MpsGroupConv2D;
}

}  // namespace gfx_plugin
}  // namespace ov
