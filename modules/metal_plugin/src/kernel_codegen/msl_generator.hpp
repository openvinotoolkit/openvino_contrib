// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "kernel_codegen/kernel_ir.hpp"

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_elementwise_add(const KernelOp& op);
std::string generate_msl_for_matmul(const KernelOp& op);

}  // namespace metal_plugin
}  // namespace ov
