// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/core/node.hpp"
#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> create_metal_stage(const std::shared_ptr<const ov::Node>& node,
                                             void* device,
                                             void* queue);
std::unique_ptr<GpuStage> create_vulkan_stage(const std::shared_ptr<const ov::Node>& node,
                                              void* device,
                                              void* queue);

}  // namespace gfx_plugin
}  // namespace ov
