// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_ir/gfx_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource &
opencl_generated_eltwise_compare_f32_kernel_source() noexcept;
const GfxKernelSource &
opencl_generated_eltwise_compare_broadcast_f32_kernel_source() noexcept;
const GfxKernelSource &
opencl_generated_eltwise_select_f32_kernel_source() noexcept;
const GfxKernelSource &
opencl_generated_eltwise_select_broadcast_f32_kernel_source() noexcept;
const GfxKernelSource &
opencl_generated_eltwise_select_f16_dynamic_kernel_source() noexcept;

} // namespace gfx_plugin
} // namespace ov
