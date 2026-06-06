// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "backends/opencl/runtime/opencl_api.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage>
create_opencl_source_stage(std::shared_ptr<OpenClRuntimeContext> context,
                           RuntimeStageExecutableDescriptor descriptor,
                           GfxOpenClSourceArtifact artifact,
                           std::shared_ptr<const ov::Node> source_node = {});

} // namespace gfx_plugin
} // namespace ov
