// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/softmax_f32_dynamic_kernel.hpp"

#include "kernel_ir/opencl_kernels/generated/softmax_f32_dynamic_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource &
opencl_generated_softmax_f32_dynamic_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/softmax_f32_dynamic_static_rank", "opencl",
      "gfx_opencl_generated_softmax_dynamic_f32",
      GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedSoftmaxDynamicF32KernelSource};
  return kKernel;
}

} // namespace gfx_plugin
} // namespace ov
