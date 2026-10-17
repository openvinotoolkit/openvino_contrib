// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/eltwise_compare_select_kernel.hpp"

#include "kernel_ir/opencl_kernels/generated/eltwise_compare_select_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource &
opencl_generated_eltwise_compare_f32_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/eltwise_compare_f32", "opencl",
      "gfx_opencl_generated_eltwise_compare_f32",
      GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedEltwiseCompareSelectKernelSource};
  return kKernel;
}

const GfxKernelSource &
opencl_generated_eltwise_compare_broadcast_f32_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/eltwise_compare_broadcast_f32", "opencl",
      "gfx_opencl_generated_eltwise_compare_broadcast_f32",
      GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedEltwiseCompareSelectKernelSource};
  return kKernel;
}

const GfxKernelSource &
opencl_generated_eltwise_select_f32_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/eltwise_select_f32", "opencl",
      "gfx_opencl_generated_eltwise_select_f32",
      GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedEltwiseCompareSelectKernelSource};
  return kKernel;
}

const GfxKernelSource &
opencl_generated_eltwise_select_broadcast_f32_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/eltwise_select_broadcast_f32", "opencl",
      "gfx_opencl_generated_eltwise_select_broadcast_f32",
      GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedEltwiseCompareSelectKernelSource};
  return kKernel;
}

const GfxKernelSource &
opencl_generated_eltwise_select_f16_dynamic_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/eltwise_select_f16_dynamic", "opencl",
      "gfx_opencl_generated_eltwise_select_f16_dynamic",
      GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedEltwiseCompareSelectKernelSource};
  return kKernel;
}

} // namespace gfx_plugin
} // namespace ov
