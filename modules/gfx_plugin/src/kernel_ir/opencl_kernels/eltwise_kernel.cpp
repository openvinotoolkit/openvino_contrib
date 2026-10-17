// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/eltwise_kernel.hpp"

#include "kernel_ir/opencl_kernels/generated/eltwise_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource& opencl_generated_eltwise_kernel_source() noexcept {
    static constexpr GfxKernelSource kKernel{
        "opencl/generated/eltwise",
        "opencl",
        "gfx_opencl_generated_eltwise_binary_f32",
        GfxKernelSourceLanguage::OpenCL,
        detail::kOpenClGeneratedEltwiseKernelSource};
    return kKernel;
}

}  // namespace gfx_plugin
}  // namespace ov
