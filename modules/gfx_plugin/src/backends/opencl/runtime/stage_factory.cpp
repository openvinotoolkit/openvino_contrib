// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/runtime/stage_factory.hpp"

#include "openvino/core/except.hpp"
#include "plugin/stateful_stage.hpp"
#include "runtime/execution_dispatcher.hpp"
#include "runtime/view_only_stage.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> create_opencl_stage(const std::shared_ptr<const ov::Node>& node,
                                              void*,
                                              void*) {
    if (auto stateful = create_stateful_stage(node)) {
        return stateful;
    }
    OPENVINO_THROW("GFX OpenCL: runtime stage materialization requires a compiler-owned executable descriptor and artifact payload for op ",
                   node ? node->get_type_name() : "<null>",
                   ". Add or fix the route in compiler manifest/artifact packaging; runtime is not allowed to resolve OpenCL source artifacts.");
}

std::unique_ptr<GpuStage> create_opencl_stage(
    const std::shared_ptr<const ov::Node>& node,
    const RuntimeStageExecutableDescriptor* descriptor) {
    if (auto stateful = create_stateful_stage(node)) {
        return stateful;
    }
    if (auto view = create_view_only_stage(node, descriptor)) {
        return view;
    }
    return create_opencl_stage(node, nullptr, nullptr);
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
