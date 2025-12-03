// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "kernel_ir/kernel_ir_common.hpp"

namespace ov {
namespace metal_plugin {

namespace msl {
std::string activation_expr(ActivationKind kind, float alpha);
}

}  // namespace metal_plugin
}  // namespace ov

