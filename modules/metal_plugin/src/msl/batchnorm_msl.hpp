// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "kernel_ir/kernel_ir_common.hpp"

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_batchnorm2d(const KernelOp& op);

}  // namespace metal_plugin
}  // namespace ov

