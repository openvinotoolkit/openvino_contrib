// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/runtime/stage_factory.hpp"

#include "openvino/core/except.hpp"
#include "runtime/execution_dispatcher.hpp"
#include "runtime/stateful_stage.hpp"
#include "runtime/view_only_stage.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> create_opencl_stage(const RuntimeStageMaterializationContext& context,
                                              void*,
                                              void*) {
    const auto& descriptor = context.require_descriptor();
    if (auto stateful = create_stateful_stage(descriptor)) {
        return stateful;
    }
    if (auto view = create_view_only_stage(descriptor)) {
        return view;
    }
    OPENVINO_THROW("GFX OpenCL: runtime stage materialization requires a compiler-owned executable descriptor and artifact payload for op ",
                   context.op_type_name(),
                   ". Add or fix the route in compiler manifest/artifact packaging; runtime is not allowed to resolve OpenCL source artifacts.");
}

void ensure_opencl_stage_factory_registered() {
    static const bool registered = GpuStageFactory::register_factory(GpuBackend::OpenCL, &create_opencl_stage);
    (void)registered;
}

namespace {
const bool kRegistered = (ensure_opencl_stage_factory_registered(), true);
}  // namespace

}  // namespace gfx_plugin
}  // namespace ov
