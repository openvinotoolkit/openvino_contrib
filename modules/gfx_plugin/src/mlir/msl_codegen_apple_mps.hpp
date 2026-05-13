// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string_view>

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "mlir/gfx_mpsrt_source_plan.hpp"
#include "openvino/core/node.hpp"
#include "runtime/gfx_activation.hpp"
#include "runtime/gfx_bias.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
namespace gfx_plugin {

class GpuBufferManager;

GfxMpsrtKernelSourcePlan configure_apple_mps_vendor_kernel_source_plan_for_node(
    KernelSource source, const std::shared_ptr<const ov::Node> &node,
    const GpuBufferManager *buffer_manager, std::string_view stage_type,
    bool has_bias, bool has_activation, bool has_batchnorm,
    ActivationKind activation, const BiasParams *bias_params = nullptr,
    const GfxStageRuntimeTraits &traits = GfxStageRuntimeTraits{});

} // namespace gfx_plugin
} // namespace ov
