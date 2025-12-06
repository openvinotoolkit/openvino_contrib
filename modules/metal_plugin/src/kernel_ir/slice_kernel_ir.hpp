// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_ir_common.hpp"

namespace ov {
namespace metal_plugin {

MetalKernelIR build_kernel_ir_for_slice(const std::shared_ptr<const ov::Model>& model);

// Utility to build a single Slice KernelOp from raw parameters (used for Split expansion).
KernelOp make_slice_op(const std::vector<int64_t>& in_shape,
                       const std::vector<int64_t>& starts,
                       const std::vector<int64_t>& steps,
                       const std::vector<int64_t>& out_shape,
                       ov::element::Type et,
                       KernelTensor& in_tensor,
                       KernelTensor& out_tensor);

}  // namespace metal_plugin
}  // namespace ov
