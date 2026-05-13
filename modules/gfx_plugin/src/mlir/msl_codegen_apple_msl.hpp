// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <optional>
#include <string_view>

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "mlir/gfx_mpsrt_source_plan.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {

class GpuBufferManager;

GfxMpsrtKernelSourcePlan configure_apple_metal_msl_kernel_source_plan(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node,
    const GpuBufferManager *buffer_manager, std::string_view stage_type,
    const ov::element::Type &storage_type, bool has_runtime_slice_params,
    const std::optional<ov::Shape> &runtime_input_shape, bool has_bias,
    bool has_activation, bool has_batchnorm);

} // namespace gfx_plugin
} // namespace ov
