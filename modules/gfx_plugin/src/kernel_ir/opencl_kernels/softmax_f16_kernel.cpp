// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/softmax_f16_kernel.hpp"

#include "kernel_ir/opencl_kernels/generated/softmax_f16_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource &opencl_generated_softmax_f16_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/softmax_f16", "opencl",
      "gfx_opencl_generated_softmax_f16", GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedSoftmaxF16KernelSource};
  return kKernel;
}

} // namespace gfx_plugin
} // namespace ov
