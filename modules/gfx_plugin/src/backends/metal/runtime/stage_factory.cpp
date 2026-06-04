// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/stage_factory.hpp"

#include "backends/metal/runtime/metal_executor.hpp"
#include "backends/metal/runtime/mpsrt_vendor_primitive_stage.hpp"
#include "runtime/execution_dispatcher.hpp"
#include "runtime/stateful_stage.hpp"
#include "runtime/view_only_stage.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage>
create_metal_stage(const std::shared_ptr<const ov::Node> &node,
                   const RuntimeStageExecutableDescriptor *descriptor,
                   void *device, void *queue) {
  if (auto stateful = create_stateful_stage(node, descriptor)) {
    return stateful;
  }
  if (auto view = create_view_only_stage(node, descriptor)) {
    return view;
  }
  if (descriptor && is_metal_mpsrt_vendor_primitive_descriptor(*descriptor)) {
    return create_metal_mpsrt_vendor_primitive_stage(node, device, queue,
                                                     *descriptor);
  }
  OPENVINO_ASSERT(
      descriptor,
      "GFX Metal: runtime stage materialization requires a compiler-owned "
      "executable descriptor and artifact payload for op ",
      node ? node->get_type_name() : "<null>",
      ". Add or fix the route in compiler manifest/artifact packaging; runtime "
      "is not allowed to infer MLIR layout or rebuild source-plan metadata.");
  return std::make_unique<MetalStage>(node, device, queue, descriptor);
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
