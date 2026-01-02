// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/stage_factory.hpp"

#include "openvino/core/except.hpp"
#include "runtime/execution_dispatcher.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> create_metal_stage(const std::shared_ptr<const ov::Node>&,
                                             void*,
                                             void*) {
    OPENVINO_THROW("GFX Metal backend is not available in this build");
}

void ensure_metal_stage_factory_registered() {
    static const bool registered = GpuStageFactory::register_factory(GpuBackend::Metal, &create_metal_stage);
    (void)registered;
}

namespace {
const bool kRegistered = (ensure_metal_stage_factory_registered(), true);
}  // namespace

}  // namespace gfx_plugin
}  // namespace ov
