// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include "kernel_ir/kernel_ir_common.hpp"

namespace ov {
namespace metal_plugin {

// Generate MSL source for variadic Concat (axis-aware, contiguous copy on GPU).
// Single kernel copies one input slice into the correct output offset; runtime
// dispatches the kernel once per input tensor with different params.
std::string generate_msl_for_concat(const KernelOp& op);

}  // namespace metal_plugin
}  // namespace ov

