// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/binary_f32_kernel.hpp"

#include "kernel_ir/opencl_kernels/generated/binary_f32_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource& opencl_baseline_binary_f32_kernel_source() noexcept {
    static constexpr GfxKernelSource kKernel{
        "opencl/baseline/eltwise_binary_f32",
        "opencl",
        "gfx_opencl_baseline_binary_f32",
        GfxKernelSourceLanguage::OpenCL,
        detail::kOpenClBaselineBinaryF32KernelSource};
    return kKernel;
}

}  // namespace gfx_plugin
}  // namespace ov
