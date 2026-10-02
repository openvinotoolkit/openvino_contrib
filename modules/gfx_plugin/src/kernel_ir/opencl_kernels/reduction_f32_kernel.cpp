// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/reduction_f32_kernel.hpp"

#include "kernel_ir/opencl_kernels/generated/reduction_f32_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource &opencl_generated_reduction_f32_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/reduction_f32", "opencl",
      "gfx_opencl_generated_reduction_f32", GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedReductionF32KernelSource};
  return kKernel;
}

} // namespace gfx_plugin
} // namespace ov
