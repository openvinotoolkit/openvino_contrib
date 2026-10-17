// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/conv2d_kernel.hpp"

#include "kernel_ir/opencl_kernels/generated/conv2d_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource &opencl_generated_conv2d_f32_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/conv2d_f32", "opencl",
      "gfx_opencl_generated_conv2d_f32", GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedConv2DF32KernelSource};
  return kKernel;
}

const GfxKernelSource &
opencl_generated_group_conv2d_f32_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/group_conv2d_f32", "opencl",
      "gfx_opencl_generated_group_conv2d_f32", GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedConv2DF32KernelSource};
  return kKernel;
}

} // namespace gfx_plugin
} // namespace ov
