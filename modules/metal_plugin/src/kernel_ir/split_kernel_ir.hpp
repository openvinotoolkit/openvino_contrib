// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_ir_common.hpp"

namespace ov {
namespace metal_plugin {

// Build Kernel IR for v1::Split (and v1::VariadicSplit) as a single KernelOpKind::Split
// carrying SplitDesc (axis + split sizes).
MetalKernelIR build_kernel_ir_for_split(const std::shared_ptr<const ov::Model>& model);

}  // namespace metal_plugin
}  // namespace ov

