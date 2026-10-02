// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/activation_kernel.hpp"

#include "kernel_ir/opencl_kernels/generated/activation_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource& opencl_generated_activation_kernel_source() noexcept {
    static constexpr GfxKernelSource kKernel{
        "opencl/generated/activation",
        "opencl",
        "gfx_opencl_generated_activation_f32",
        GfxKernelSourceLanguage::OpenCL,
        detail::kOpenClGeneratedActivationKernelSource};
    return kKernel;
}

}  // namespace gfx_plugin
}  // namespace ov
