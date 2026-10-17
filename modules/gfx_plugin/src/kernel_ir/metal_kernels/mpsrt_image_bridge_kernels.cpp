// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/metal_kernels/mpsrt_image_bridge_kernels.hpp"

#include "kernel_ir/metal_kernels/generated/mpsrt_buffer_to_image_f16_source.hpp"
#include "kernel_ir/metal_kernels/generated/mpsrt_buffer_to_image_f32_source.hpp"
#include "kernel_ir/metal_kernels/generated/mpsrt_image_to_buffer_f16_source.hpp"
#include "kernel_ir/metal_kernels/generated/mpsrt_image_to_buffer_f32_source.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

const GfxKernelSource& mpsrt_image_bridge_kernel_source(MpsrtImageBridgeKernelKind kind) {
    static constexpr GfxKernelSource kBufferToImageF32{
        "mpsrt_bridge_buffer_to_image_f32",
        "apple_msl",
        "gfx_mpsrt_buffer_to_image_f32",
        GfxKernelSourceLanguage::MetalShadingLanguage,
        detail::kMpsrtBufferToImageF32Source};
    static constexpr GfxKernelSource kBufferToImageF16{
        "mpsrt_bridge_buffer_to_image_f16",
        "apple_msl",
        "gfx_mpsrt_buffer_to_image_f16",
        GfxKernelSourceLanguage::MetalShadingLanguage,
        detail::kMpsrtBufferToImageF16Source};
    static constexpr GfxKernelSource kImageToBufferF32{
        "mpsrt_bridge_image_to_buffer_f32",
        "apple_msl",
        "gfx_mpsrt_image_to_buffer_f32",
        GfxKernelSourceLanguage::MetalShadingLanguage,
        detail::kMpsrtImageToBufferF32Source};
    static constexpr GfxKernelSource kImageToBufferF16{
        "mpsrt_bridge_image_to_buffer_f16",
        "apple_msl",
        "gfx_mpsrt_image_to_buffer_f16",
        GfxKernelSourceLanguage::MetalShadingLanguage,
        detail::kMpsrtImageToBufferF16Source};

    switch (kind) {
    case MpsrtImageBridgeKernelKind::BufferToImageF32:
        return kBufferToImageF32;
    case MpsrtImageBridgeKernelKind::BufferToImageF16:
        return kBufferToImageF16;
    case MpsrtImageBridgeKernelKind::ImageToBufferF32:
        return kImageToBufferF32;
    case MpsrtImageBridgeKernelKind::ImageToBufferF16:
        return kImageToBufferF16;
    }
    OPENVINO_THROW("GFX MPSRT: unknown image bridge kernel kind");
}

}  // namespace gfx_plugin
}  // namespace ov
