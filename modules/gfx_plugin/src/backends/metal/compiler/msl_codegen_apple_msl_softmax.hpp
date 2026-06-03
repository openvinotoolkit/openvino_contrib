// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <optional>
#include <string_view>

#include "mlir/IR/BuiltinOps.h"
#include "backends/metal/compiler/msl_codegen_apple_msl_dispatch.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace gfx_plugin {

struct SoftmaxMslKernelDescriptor {
  std::string_view kernel_unit_id;
  std::string_view entry_point;
  bool log_softmax = false;
};

std::optional<SoftmaxMslKernelDescriptor>
softmax_msl_kernel_descriptor(const std::shared_ptr<const ov::Node> &node);

GfxMslGeneratedKernelSourcePlan
make_softmax_msl_kernel_source_plan(const std::shared_ptr<const ov::Node> &node,
                                    mlir::ModuleOp module = {});

GfxMslGeneratedKernelSourcePlan
make_softmax_runtime_params_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node, mlir::ModuleOp module = {});

} // namespace gfx_plugin
} // namespace ov
