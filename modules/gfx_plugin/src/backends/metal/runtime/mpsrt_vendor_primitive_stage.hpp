// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "backends/metal/runtime/metal_memory.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gpu_stage.hpp"
#include "runtime/gpu_types.hpp"

namespace ov {
namespace gfx_plugin {

bool is_metal_mpsrt_vendor_primitive_descriptor(
    const RuntimeStageExecutableDescriptor& descriptor) noexcept;

std::unique_ptr<GpuStage> create_metal_mpsrt_vendor_primitive_stage(
    MetalDeviceHandle device,
    MetalCommandQueueHandle queue,
    const RuntimeStageExecutableDescriptor& descriptor);

}  // namespace gfx_plugin
}  // namespace ov
