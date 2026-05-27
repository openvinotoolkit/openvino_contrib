// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/core/node.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> create_opencl_stage(const std::shared_ptr<const ov::Node>& node,
                                              void* device,
                                              void* queue);
std::unique_ptr<GpuStage> create_opencl_stage(
    const std::shared_ptr<const ov::Node>& node,
    const RuntimeStageExecutableDescriptor* descriptor);
void ensure_opencl_stage_factory_registered();

}  // namespace gfx_plugin
}  // namespace ov
