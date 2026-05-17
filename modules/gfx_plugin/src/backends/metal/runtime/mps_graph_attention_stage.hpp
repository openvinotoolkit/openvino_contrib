// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "backends/metal/runtime/metal_memory.hpp"
#include "plugin/backend_state.hpp"
#include "runtime/gpu_stage.hpp"
#include "runtime/gpu_types.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> create_metal_vendor_attention_stage(
    VendorAttentionStageSpec spec,
    MetalDeviceHandle device,
    MetalCommandQueueHandle queue);

}  // namespace gfx_plugin
}  // namespace ov
