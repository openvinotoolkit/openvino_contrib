// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/eltwise_logical_bool_kernel.hpp"

#include "kernel_ir/opencl_kernels/generated/eltwise_logical_bool_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource &
opencl_generated_eltwise_logical_unary_bool_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/eltwise_logical_unary_bool", "opencl",
      "gfx_opencl_generated_eltwise_logical_unary_bool",
      GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedEltwiseLogicalBoolKernelSource};
  return kKernel;
}

const GfxKernelSource &
opencl_generated_eltwise_logical_binary_bool_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/eltwise_logical_binary_bool", "opencl",
      "gfx_opencl_generated_eltwise_logical_binary_bool",
      GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedEltwiseLogicalBoolKernelSource};
  return kKernel;
}

const GfxKernelSource &
opencl_generated_eltwise_logical_binary_broadcast_bool_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/eltwise_logical_binary_broadcast_bool", "opencl",
      "gfx_opencl_generated_eltwise_logical_binary_broadcast_bool",
      GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedEltwiseLogicalBoolKernelSource};
  return kKernel;
}

} // namespace gfx_plugin
} // namespace ov
