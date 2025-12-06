// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "kernel_ir/kernel_ir_common.hpp"

namespace ov {
namespace metal_plugin {

// Generate MSL source for a Split KernelOp. Produces a single kernel that
// routes elements from one input buffer to N output buffers based on the
// split sizes along the given axis. All shapes/sizes are baked into the
// generated source for simplicity.
std::string generate_split_msl(const KernelOp& op);

}  // namespace metal_plugin
}  // namespace ov

