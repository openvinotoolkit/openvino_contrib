// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_ir/gfx_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

enum class MpsrtImageBridgeKernelKind {
    BufferToImageF32,
    BufferToImageF16,
    ImageToBufferF32,
    ImageToBufferF16,
};

const GfxKernelSource& mpsrt_image_bridge_kernel_source(MpsrtImageBridgeKernelKind kind);

}  // namespace gfx_plugin
}  // namespace ov
