// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace ov {
namespace gfx_plugin {

namespace detail {

bool is_conv_tile_input_h_interior(int64_t oh_base,
                                   int64_t tile_h,
                                   int64_t stride_h,
                                   int64_t dil_h,
                                   int64_t kernel_h,
                                   int64_t pad_h,
                                   int64_t input_h);

bool is_conv_tile_input_w_interior(int64_t ow_base,
                                   int64_t tile_w,
                                   int64_t stride_w,
                                   int64_t dil_w,
                                   int64_t kernel_w,
                                   int64_t pad_w,
                                   int64_t input_w);

bool is_conv_tile_input_interior(int64_t oh_base,
                                 int64_t ow_base,
                                 int64_t tile_h,
                                 int64_t tile_w,
                                 int64_t stride_h,
                                 int64_t stride_w,
                                 int64_t dil_h,
                                 int64_t dil_w,
                                 int64_t kernel_h,
                                 int64_t kernel_w,
                                 int64_t pad_h,
                                 int64_t pad_w,
                                 int64_t input_h,
                                 int64_t input_w);

}  // namespace detail

// Lower linalg::Conv2DNchwFchwOp to explicit scf.parallel + scf.for loops.
// Intended to stabilize Vulkan SPIR-V parallel path for Conv2D.
void run_conv2d_parallel_lowering(mlir::ModuleOp module);

}  // namespace gfx_plugin
}  // namespace ov
