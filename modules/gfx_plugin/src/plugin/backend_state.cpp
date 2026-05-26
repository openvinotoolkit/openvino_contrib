// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/backend_state.hpp"

#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> BackendState::create_stage(
    const std::shared_ptr<const ov::Node>& node,
    const RuntimeStageExecutableDescriptor* /*descriptor*/) const {
    return create_stage(node);
}

}  // namespace gfx_plugin
}  // namespace ov
