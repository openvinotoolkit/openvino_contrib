// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/msl_codegen_apple_msl_dispatch.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {

std::string
generate_static_msl_for_slice(const std::shared_ptr<const ov::Node> &node,
                              const ov::element::Type &storage_type);

GfxMslGeneratedKernelSourcePlan make_direct_static_slice_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node,
    const ov::element::Type &storage_type, mlir::ModuleOp module = {});

} // namespace gfx_plugin
} // namespace ov
