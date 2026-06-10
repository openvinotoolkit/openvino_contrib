// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_ir/gfx_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource &opencl_generated_range_f32_kernel_source() noexcept;
const GfxKernelSource &opencl_generated_range_f16_kernel_source() noexcept;
const GfxKernelSource &opencl_generated_range_i64_kernel_source() noexcept;
const GfxKernelSource &opencl_generated_range_i64_unit_kernel_source() noexcept;

} // namespace gfx_plugin
} // namespace ov
