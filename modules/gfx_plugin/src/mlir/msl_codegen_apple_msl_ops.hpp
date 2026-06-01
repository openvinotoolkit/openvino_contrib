// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <optional>

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "mlir/msl_codegen_apple_msl_dispatch.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {

std::optional<KernelSource> make_apple_metal_slice_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node,
    const ov::element::Type &storage_type, bool has_runtime_slice_params);
std::optional<KernelSource> make_apple_metal_convolution_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node);
std::optional<KernelSource> make_apple_metal_matmul_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node);
std::optional<KernelSource>
make_apple_metal_llm_kernel_source(KernelSource source,
                                   const std::shared_ptr<const ov::Node> &node);
std::optional<KernelSource> make_apple_metal_softmax_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node,
    const std::optional<ov::Shape> &runtime_input_shape);
std::optional<KernelSource> make_apple_metal_unary_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node);
std::optional<KernelSource> make_apple_metal_elementwise_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node);
std::optional<KernelSource> make_apple_metal_layout_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node);
GfxMslGeneratedKernelSourcePlan make_transpose_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node,
    mlir::ModuleOp module = {});
std::optional<KernelSource> make_apple_metal_concat_split_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node);
std::optional<KernelSource> make_apple_metal_convert_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node);
std::optional<KernelSource> make_apple_metal_reduction_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node);
std::optional<KernelSource> make_apple_metal_topk_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node);
std::optional<KernelSource> make_apple_metal_data_movement_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node);
std::optional<KernelSource> make_apple_metal_shape_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node);

} // namespace gfx_plugin
} // namespace ov
