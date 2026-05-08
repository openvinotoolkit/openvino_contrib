// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {

bool configure_apple_metal_data_movement_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node);

bool configure_apple_metal_msl_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node,
    std::string_view stage_type, const ov::element::Type &storage_type,
    bool has_runtime_slice_params,
    const std::optional<ov::Shape> &runtime_input_shape = std::nullopt);

bool annotate_apple_msl_custom_kernel_binding(
    mlir::ModuleOp module, std::string_view stage_type,
    std::string_view entry_point, std::vector<int32_t> scalar_args = {});

void require_apple_msl_custom_kernel_binding(
    mlir::ModuleOp module, std::string_view stage_type,
    std::string_view entry_point, std::vector<int32_t> scalar_args = {});

size_t infer_apple_msl_custom_kernel_arg_count(
    mlir::ModuleOp module, size_t fallback, std::string_view entry_point = {});

} // namespace gfx_plugin
} // namespace ov
