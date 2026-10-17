// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/shapeof_kernel.hpp"

#include "kernel_ir/opencl_kernels/generated/shapeof_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource &opencl_generated_shapeof_i32_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/shapeof_i32", "opencl",
      "gfx_opencl_generated_shapeof_i32", GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedShapeOfKernelSource};
  return kKernel;
}

const GfxKernelSource &opencl_generated_shapeof_i64_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/shapeof_i64", "opencl",
      "gfx_opencl_generated_shapeof_i64", GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedShapeOfKernelSource};
  return kKernel;
}

} // namespace gfx_plugin
} // namespace ov
