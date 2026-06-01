// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/stage_factory.hpp"

#include "backends/metal/runtime/metal_executor.hpp"
#include "backends/metal/runtime/mpsrt_vendor_primitive_stage.hpp"
#include "plugin/stateful_stage.hpp"
#include "runtime/execution_dispatcher.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> create_metal_stage(const std::shared_ptr<const ov::Node>& node,
                                             const RuntimeStageExecutableDescriptor* descriptor,
                                             void* device,
                                             void* queue) {
    if (auto stateful = create_stateful_stage(node)) {
        return stateful;
    }
    if (descriptor && is_metal_mpsrt_vendor_primitive_descriptor(*descriptor)) {
        return create_metal_mpsrt_vendor_primitive_stage(node, device, queue, *descriptor);
    }
    return std::make_unique<MetalStage>(node, device, queue, descriptor);
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
