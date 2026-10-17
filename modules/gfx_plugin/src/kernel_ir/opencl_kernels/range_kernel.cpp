// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/range_kernel.hpp"

#include "kernel_ir/opencl_kernels/generated/range_f16_kernel_source.hpp"
#include "kernel_ir/opencl_kernels/generated/range_f32_kernel_source.hpp"
#include "kernel_ir/opencl_kernels/generated/range_i64_kernel_source.hpp"
#include "kernel_ir/opencl_kernels/generated/range_i64_unit_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource &opencl_generated_range_f32_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/range_f32", "opencl", "gfx_opencl_generated_range_f32",
      GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedRangeF32KernelSource};
  return kKernel;
}

const GfxKernelSource &opencl_generated_range_f16_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/range_f16", "opencl", "gfx_opencl_generated_range_f16",
      GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedRangeF16KernelSource};
  return kKernel;
}

const GfxKernelSource &opencl_generated_range_i64_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/range_i64", "opencl", "gfx_opencl_generated_range_i64",
      GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedRangeI64KernelSource};
  return kKernel;
}

const GfxKernelSource &
opencl_generated_range_i64_unit_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/range_i64_unit_dynamic", "opencl",
      "gfx_opencl_generated_range_i64_unit", GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedRangeI64UnitKernelSource};
  return kKernel;
}

} // namespace gfx_plugin
} // namespace ov
