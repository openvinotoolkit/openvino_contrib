// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "kernel_ir/kernel_ir_common.hpp"

namespace ov {
class Node;

namespace metal_plugin {

MetalKernelIR build_kernel_ir_for_unary(const std::shared_ptr<const ov::Node>& node,
                                        ActivationKind kind,
                                        float alpha);

}  // namespace metal_plugin
}  // namespace ov

