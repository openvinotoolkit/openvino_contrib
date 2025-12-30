// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_stage_factory.hpp"

#include "backends/metal/runtime/op_factory.hpp"
#include "backends/metal/runtime/metal_executor.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> create_metal_stage(const std::shared_ptr<const ov::Node>& node,
                                             void* device,
                                             void* queue) {
    auto op = MetalOpFactory::create(node, device, queue);
    if (!op) {
        return nullptr;
    }
    return std::make_unique<MetalStage>(std::move(op));
}

}  // namespace gfx_plugin
}  // namespace ov
