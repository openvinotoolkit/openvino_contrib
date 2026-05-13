// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <optional>

#include "kernel_ir/gfx_codegen_desc.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "openvino/core/node.hpp"
#include "runtime/gfx_activation.hpp"
#include "llvm/ADT/StringRef.h"

namespace ov {
namespace gfx_plugin {

std::optional<EltwiseKind> eltwise_kind_from_node(const ov::Node &node);
std::optional<ReduceKind> reduce_kind_from_node(const ov::Node &node);
std::optional<ActivationKind>
unary_activation_kind_from_node(const ov::Node &node);
std::optional<ActivationKind>
activation_kind_from_module_attr(mlir::ModuleOp module,
                                 llvm::StringRef attr_name);

} // namespace gfx_plugin
} // namespace ov
