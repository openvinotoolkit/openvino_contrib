// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/gfx_apple_stage_pipeline.hpp"
#include "mlir/gfx_apple_vendor_descriptors.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "openvino/core/node.hpp"
#include "runtime/gfx_mpsrt_abi.hpp"
#include "runtime/gfx_stage_policy.hpp"

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
    GfxMpsrtConv2DAbiDesc conv_desc{};
    if (!gfx_apple_make_mps_conv2d_desc(node, conv_desc, has_activation, activation)) {
        return GfxConvMpsrtLoweringKind::None;
    }

    const auto materialized =
        materialize_apple_mps_conv2d_program(module,
                                             plan,
                                             stage_type,
                                             conv_desc,
                                             {GfxKernelBufferRole::TensorInput,
                                              GfxKernelBufferRole::ConstTensor});
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
