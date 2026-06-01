// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/tile_kernel.hpp"

#include "kernel_ir/opencl_kernels/generated/tile_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource &opencl_generated_tile_f32_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/tile_f32", "opencl", "gfx_opencl_generated_tile_f32",
      GfxKernelSourceLanguage::OpenCL, detail::kOpenClGeneratedTileKernelSource};
  return kKernel;
}

const GfxKernelSource &
opencl_generated_tile_dynamic_f32_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/tile_dynamic_f32", "opencl",
      "gfx_opencl_generated_tile_dynamic_f32", GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedTileKernelSource};
  return kKernel;
}

const GfxKernelSource &opencl_generated_tile_f16_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/tile_f16", "opencl", "gfx_opencl_generated_tile_f16",
      GfxKernelSourceLanguage::OpenCL, detail::kOpenClGeneratedTileKernelSource};
  return kKernel;
}

const GfxKernelSource &
opencl_generated_tile_dynamic_f16_kernel_source() noexcept {
  static constexpr GfxKernelSource kKernel{
      "opencl/generated/tile_dynamic_f16", "opencl",
      "gfx_opencl_generated_tile_dynamic_f16", GfxKernelSourceLanguage::OpenCL,
      detail::kOpenClGeneratedTileKernelSource};
  return kKernel;
}

} // namespace gfx_plugin
} // namespace ov
