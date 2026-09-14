// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/matmul_f32_kernel.hpp"

#include "kernel_ir/opencl_kernels/generated/matmul_f32_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource& opencl_generated_matmul_f32_kernel_source() noexcept {
    static constexpr GfxKernelSource kKernel{
        "opencl/generated/matmul_f32",
        "opencl",
        "gfx_opencl_generated_matmul_f32",
        GfxKernelSourceLanguage::OpenCL,
        detail::kOpenClGeneratedMatMulF32KernelSource};
    return kKernel;
}

}  // namespace gfx_plugin
}  // namespace ov
