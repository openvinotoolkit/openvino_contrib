// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_stage_factory.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> create_vulkan_stage(const std::shared_ptr<const ov::Node>&,
                                              void*,
                                              void*) {
    OPENVINO_THROW("GFX Vulkan backend is not available in this build");
}

}  // namespace gfx_plugin
}  // namespace ov
