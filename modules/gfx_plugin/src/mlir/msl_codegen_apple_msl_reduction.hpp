// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <optional>
#include <string_view>

#include "kernel_ir/gfx_codegen_desc.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/msl_codegen_apple_msl_dispatch.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace gfx_plugin {

std::optional<ReduceKind>
reduction_kind_from_node(const std::shared_ptr<const ov::Node> &node);
uint32_t reduction_kernel_op_code(ReduceKind kind) noexcept;
bool reduction_kind_is_logical(ReduceKind kind) noexcept;
std::string_view reduction_msl_kernel_unit_id(ReduceKind kind) noexcept;
std::string_view reduction_msl_kernel_entry_point(ReduceKind kind) noexcept;

GfxMslGeneratedKernelSourcePlan make_reduction_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node, mlir::ModuleOp module = {});

} // namespace gfx_plugin
} // namespace ov
