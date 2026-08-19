// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_ir/gfx_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource &metal_generated_reduction_f32_kernel_source();
const GfxKernelSource &metal_generated_reduction_logical_bool_kernel_source();

} // namespace gfx_plugin
} // namespace ov
