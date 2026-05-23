// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "backends/opencl/runtime/opencl_api.hpp"
#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> create_opencl_source_stage(
    const std::shared_ptr<const ov::Node>& node,
    std::shared_ptr<OpenClRuntimeContext> context);

}  // namespace gfx_plugin
}  // namespace ov
