// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/metal_kernels/softmax_kernels.hpp"

#include "kernel_ir/metal_kernels/generated/logsoftmax_f16_kernel_source.hpp"
#include "kernel_ir/metal_kernels/generated/logsoftmax_f32_kernel_source.hpp"
#include "kernel_ir/metal_kernels/generated/softmax_f16_kernel_source.hpp"
#include "kernel_ir/metal_kernels/generated/softmax_f32_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource &metal_generated_softmax_f32_kernel_source() {
  static constexpr GfxKernelSource kKernel{
      "metal/generated/softmax_f32", "metal", "gfx_metal_generated_softmax_f32",
      GfxKernelSourceLanguage::MetalShadingLanguage,
      detail::kMetalGeneratedSoftmaxF32KernelSource};
  return kKernel;
}

const GfxKernelSource &metal_generated_softmax_f16_kernel_source() {
  static constexpr GfxKernelSource kKernel{
      "metal/generated/softmax_f16", "metal", "gfx_metal_generated_softmax_f16",
      GfxKernelSourceLanguage::MetalShadingLanguage,
      detail::kMetalGeneratedSoftmaxF16KernelSource};
  return kKernel;
}

const GfxKernelSource &metal_generated_logsoftmax_f32_kernel_source() {
  static constexpr GfxKernelSource kKernel{
      "metal/generated/logsoftmax_f32", "metal",
      "gfx_metal_generated_logsoftmax_f32",
      GfxKernelSourceLanguage::MetalShadingLanguage,
      detail::kMetalGeneratedLogSoftmaxF32KernelSource};
  return kKernel;
}

const GfxKernelSource &metal_generated_logsoftmax_f16_kernel_source() {
  static constexpr GfxKernelSource kKernel{
      "metal/generated/logsoftmax_f16", "metal",
      "gfx_metal_generated_logsoftmax_f16",
      GfxKernelSourceLanguage::MetalShadingLanguage,
      detail::kMetalGeneratedLogSoftmaxF16KernelSource};
  return kKernel;
}

} // namespace gfx_plugin
} // namespace ov
