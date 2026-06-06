// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/stage_factory.hpp"

#include <memory>
#include <string>

#include "backends/metal/runtime/metal_executor.hpp"
#include "backends/metal/runtime/mpsrt_vendor_primitive_stage.hpp"
#include "runtime/execution_dispatcher.hpp"
#include "runtime/stateful_stage.hpp"
#include "runtime/view_only_stage.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage>
create_metal_stage(const RuntimeStageMaterializationContext &context,
                   void *device, void *queue) {
  const auto &descriptor = context.require_descriptor();
  if (auto stateful = create_stateful_stage(descriptor)) {
    return stateful;
  }
  if (auto view = create_view_only_stage(descriptor)) {
    return view;
  }
  if (is_metal_mpsrt_vendor_primitive_descriptor(descriptor)) {
    const std::string reason =
        !descriptor.temporary_source_node_bridge_reason.empty()
            ? descriptor.temporary_source_node_bridge_reason
            : "Metal MPSRT vendor primitive runtime still needs source-node "
              "bridge until vendor constants and layout metadata are "
              "descriptor-owned";
    const auto &node = context.require_source_node(reason);
    return create_metal_mpsrt_vendor_primitive_stage(node, device, queue,
                                                     descriptor);
  }
  std::shared_ptr<const ov::Node> node;
  if (descriptor.temporary_source_node_bridge_required) {
    node = context.require_source_node(
        descriptor.temporary_source_node_bridge_reason);
  }
  return std::make_unique<MetalStage>(descriptor, device, queue, node);
}

void ensure_metal_stage_factory_registered() {
  static const bool registered =
      GpuStageFactory::register_factory(GpuBackend::Metal, &create_metal_stage);
  (void)registered;
}

namespace {
const bool kRegistered = (ensure_metal_stage_factory_registered(), true);
} // namespace

} // namespace gfx_plugin
} // namespace ov
