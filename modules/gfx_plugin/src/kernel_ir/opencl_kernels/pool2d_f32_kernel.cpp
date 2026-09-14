// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/pool2d_f32_kernel.hpp"

#include "kernel_ir/opencl_kernels/generated/pool2d_f32_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource& opencl_generated_pool2d_f32_kernel_source() noexcept {
    static constexpr GfxKernelSource kKernel{
        "opencl/generated/pool2d_f32",
        "opencl",
        "gfx_opencl_generated_pool2d_f32",
        GfxKernelSourceLanguage::OpenCL,
        detail::kOpenClGeneratedPool2DF32KernelSource};
    return kKernel;
}

}  // namespace gfx_plugin
}  // namespace ov
