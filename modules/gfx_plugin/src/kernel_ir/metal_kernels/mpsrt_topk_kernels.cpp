// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/metal_kernels/mpsrt_topk_kernels.hpp"

#include "kernel_ir/metal_kernels/generated/mpsrt_topk_pack_u32_to_i64_source.hpp"
#include "kernel_ir/metal_kernels/generated/mpsrt_topk_stable_i64_indices_f16_source.hpp"
#include "kernel_ir/metal_kernels/generated/mpsrt_topk_stable_i64_indices_f32_source.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource& mpsrt_topk_pack_u32_to_i64_kernel_source() {
    static constexpr GfxKernelSource kKernel{
        "mpsrt_topk_pack_u32_to_i64",
        "apple_msl",
        "gfx_mpsrt_topk_pack_u32_to_i64",
        GfxKernelSourceLanguage::MetalShadingLanguage,
        detail::kMpsrtTopKPackU32ToI64Source};
    return kKernel;
}

const GfxKernelSource& mpsrt_topk_stable_i64_indices_kernel_source(MpsrtTopKValueType value_type) {
    static constexpr GfxKernelSource kF32Kernel{
        "mpsrt_topk_stable_i64_indices_f32",
        "apple_msl",
        "gfx_mpsrt_topk_stable_i64_indices",
        GfxKernelSourceLanguage::MetalShadingLanguage,
        detail::kMpsrtTopKStableI64IndicesF32Source};
    static constexpr GfxKernelSource kF16Kernel{
        "mpsrt_topk_stable_i64_indices_f16",
        "apple_msl",
        "gfx_mpsrt_topk_stable_i64_indices",
        GfxKernelSourceLanguage::MetalShadingLanguage,
        detail::kMpsrtTopKStableI64IndicesF16Source};

    switch (value_type) {
    case MpsrtTopKValueType::F32:
        return kF32Kernel;
    case MpsrtTopKValueType::F16:
        return kF16Kernel;
    }
    OPENVINO_THROW("GFX MPSRT: unknown TopK value type");
}

}  // namespace gfx_plugin
}  // namespace ov
