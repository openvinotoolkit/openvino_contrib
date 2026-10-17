// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "runtime/executable_descriptor.hpp"
#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

bool is_compiler_owned_view_stage_descriptor(
    const RuntimeStageExecutableDescriptor& descriptor) noexcept;

std::unique_ptr<GpuStage> create_view_only_stage(
    const RuntimeStageExecutableDescriptor& descriptor);

}  // namespace gfx_plugin
}  // namespace ov
