// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "runtime/gfx_mpsrt_abi.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
namespace gfx_plugin {

enum class GfxConvMpsrtLoweringKind {
    None,
    MpsConv2D,
    MpsGroupConv2D,
};

inline std::string gfx_mpsrt_canonical_conv_stage_type(const std::shared_ptr<const ov::Node>& node,
                                                       std::string_view fallback_stage_type) {
    if (ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
        return "Convolution";
    }
    if (ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
        return "GroupConvolution";
    }
    if (fallback_stage_type == "GroupConv2D") {
        return "GroupConvolution";
    }
    return std::string(fallback_stage_type);
}

namespace detail {

inline bool gfx_mpsrt_copy_conv_spatial_attrs(const ov::Strides& strides,
                                              const ov::Strides& dilations,
                                              const ov::CoordinateDiff& pads_begin,
                                              const ov::CoordinateDiff& pads_end,
                                              GfxMpsrtConv2DAbiDesc& desc) {
    if (strides.size() != 2 || dilations.size() != 2 ||
        pads_begin.size() != 2 || pads_end.size() != 2) {
        return false;
    }
    if (pads_begin[0] < 0 || pads_begin[1] < 0 ||
        pads_end[0] < 0 || pads_end[1] < 0) {
        return false;
    }

    desc.strides[0] = static_cast<uint32_t>(strides[0]);
    desc.strides[1] = static_cast<uint32_t>(strides[1]);
    desc.dilations[0] = static_cast<uint32_t>(dilations[0]);
    desc.dilations[1] = static_cast<uint32_t>(dilations[1]);
    desc.pads[0] = static_cast<uint32_t>(pads_begin[0]);
    desc.pads[1] = static_cast<uint32_t>(pads_begin[1]);
    desc.pads[2] = static_cast<uint32_t>(pads_end[0]);
    desc.pads[3] = static_cast<uint32_t>(pads_end[1]);
    return true;
}

}  // namespace detail

inline bool make_mpsrt_conv2d_desc_from_node(const std::shared_ptr<const ov::Node>& node,
                                             GfxMpsrtConv2DAbiDesc& desc) {
    desc = {};
    if (!node || node->get_input_size() < 2 ||
        !node->get_input_partial_shape(0).is_static() ||
        !node->get_input_partial_shape(1).is_static()) {
        return false;
    }

    if (auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
        const auto input_shape = conv->get_input_shape(0);
        const auto weights_shape = conv->get_input_shape(1);
        if (input_shape.size() != 4 || weights_shape.size() != 4 ||
            weights_shape[1] == 0 || input_shape[1] % weights_shape[1] != 0) {
            return false;
        }
        desc.groups = static_cast<uint32_t>(input_shape[1] / weights_shape[1]);
        return detail::gfx_mpsrt_copy_conv_spatial_attrs(conv->get_strides(),
                                                         conv->get_dilations(),
                                                         conv->get_pads_begin(),
                                                         conv->get_pads_end(),
                                                         desc);
    }

    if (auto group_conv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
        const auto input_shape = group_conv->get_input_shape(0);
        const auto weights_shape = group_conv->get_input_shape(1);
        if (input_shape.size() != 4 || weights_shape.size() != 5 || weights_shape[0] == 0) {
            return false;
        }
        desc.groups = static_cast<uint32_t>(weights_shape[0]);
        return detail::gfx_mpsrt_copy_conv_spatial_attrs(group_conv->get_strides(),
                                                         group_conv->get_dilations(),
                                                         group_conv->get_pads_begin(),
                                                         group_conv->get_pads_end(),
                                                         desc);
    }

    return false;
}

inline GfxConvMpsrtLoweringKind annotate_module_with_conv_mpsrt_plan(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const std::shared_ptr<const ov::Node>& node,
    std::string_view fallback_stage_type) {
    if (!module ||
        plan.placement.domain != GfxStageBackendDomain::AppleMps ||
        !plan.placement.uses_vendor_primitive ||
        plan.placement.uses_custom_kernel) {
        return GfxConvMpsrtLoweringKind::None;
    }

    const auto stage_type = gfx_mpsrt_canonical_conv_stage_type(node, fallback_stage_type);
    const auto stage_desc = gfx_mpsrt_make_stage_desc(plan, stage_type);
    if (stage_desc.kind != GfxMpsrtStageKind::MPSConv2D &&
        stage_desc.kind != GfxMpsrtStageKind::MPSGroupConv2D) {
        return GfxConvMpsrtLoweringKind::None;
    }

    GfxMpsrtConv2DAbiDesc conv_desc{};
    if (!make_mpsrt_conv2d_desc_from_node(node, conv_desc)) {
        return GfxConvMpsrtLoweringKind::None;
    }

    annotate_module_with_mpsrt_stage_plan(module, plan, stage_type);
    annotate_module_with_mpsrt_conv2d_desc(module, conv_desc);
    return stage_desc.kind == GfxMpsrtStageKind::MPSConv2D
               ? GfxConvMpsrtLoweringKind::MpsConv2D
               : GfxConvMpsrtLoweringKind::MpsGroupConv2D;
}

}  // namespace gfx_plugin
}  // namespace ov
