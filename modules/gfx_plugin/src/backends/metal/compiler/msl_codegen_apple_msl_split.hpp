// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "kernel_ir/gfx_codegen_desc.hpp"
#include "kernel_ir/gfx_codegen_backend.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_dispatch.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {

GfxMslGeneratedKernelSourcePlan make_direct_concat_msl_kernel_source_plan(
    KernelSource source, const ConcatCodegenDesc &desc);

GfxMslGeneratedKernelSourcePlan make_concat_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node,
    mlir::ModuleOp module = {});

GfxMslGeneratedKernelSourcePlan make_split_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node,
    mlir::ModuleOp module = {});

GfxMslGeneratedKernelSourcePlan make_direct_split_msl_kernel_source_plan(
    std::string_view stage_type, const ov::element::Type &element_type,
    const ov::Shape &input_shape, const std::vector<size_t> &split_sizes,
    uint32_t axis_len, uint32_t inner_stride, mlir::ModuleOp module = {});

} // namespace gfx_plugin
} // namespace ov
