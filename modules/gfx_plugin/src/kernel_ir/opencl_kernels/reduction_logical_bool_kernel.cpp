// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/reduction_logical_bool_kernel.hpp"

#include "kernel_ir/opencl_kernels/generated/reduction_logical_bool_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource &
opencl_baseline_reduction_logical_bool_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/baseline/reduce_logical_bool", "opencl",
      "gfx_opencl_baseline_reduce_logical_bool",
      GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClBaselineReductionLogicalBoolKernelSource};
  return kKernel;
}

} // namespace gfx_plugin
} // namespace ov
