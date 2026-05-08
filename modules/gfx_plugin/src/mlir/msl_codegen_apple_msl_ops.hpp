// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <optional>

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {

bool configure_apple_metal_slice_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node,
    const ov::element::Type &storage_type, bool has_runtime_slice_params);
bool configure_apple_metal_compute_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node);
bool configure_apple_metal_softmax_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node,
    const std::optional<ov::Shape> &runtime_input_shape);
bool configure_apple_metal_pool2d_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node);
bool configure_apple_metal_unary_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node);
bool configure_apple_metal_elementwise_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node);
bool configure_apple_metal_structural_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node);

} // namespace gfx_plugin
} // namespace ov
