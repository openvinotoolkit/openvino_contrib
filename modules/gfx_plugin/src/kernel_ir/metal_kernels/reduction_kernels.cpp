// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/metal_kernels/reduction_kernels.hpp"

#include "kernel_ir/metal_kernels/generated/reduction_f32_kernel_source.hpp"
#include "kernel_ir/metal_kernels/generated/reduction_logical_bool_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource &metal_generated_reduction_f32_kernel_source() {
  static constexpr GfxKernelSource kKernel{
      "metal/generated/reduction_f32", "metal",
      "gfx_metal_generated_reduction_f32",
      GfxKernelSourceLanguage::MetalShadingLanguage,
      detail::kMetalGeneratedReductionF32KernelSource};
  return kKernel;
}

const GfxKernelSource &metal_generated_reduction_logical_bool_kernel_source() {
  static constexpr GfxKernelSource kKernel{
      "metal/generated/reduction_logical_bool", "metal",
      "gfx_metal_generated_reduction_logical_bool",
      GfxKernelSourceLanguage::MetalShadingLanguage,
      detail::kMetalGeneratedReductionLogicalBoolKernelSource};
  return kKernel;
}

} // namespace gfx_plugin
} // namespace ov
