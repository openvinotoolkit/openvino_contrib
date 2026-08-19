// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "runtime/gpu_stage.hpp"
#include "runtime/stage_materialization_context.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> create_metal_stage(const RuntimeStageMaterializationContext& context,
                                             void* device,
                                             void* queue);
void ensure_metal_stage_factory_registered();

}  // namespace gfx_plugin
}  // namespace ov
