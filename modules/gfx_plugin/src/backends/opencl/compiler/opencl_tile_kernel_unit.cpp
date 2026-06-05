// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_tile_kernel_unit.hpp"

#include "compiler/executable_bundle.hpp"
#include "compiler/lowering_planner.hpp"
#include "kernel_ir/opencl_kernels/tile_kernel.hpp"
#include "openvino/op/tile.hpp"

#include <utility>

namespace ov {
namespace gfx_plugin {
namespace compiler {

bool is_opencl_tile_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(ov::as_type_ptr<const ov::op::v0::Tile>(node));
}

namespace {

std::string support_reason_for_tile_unit(const KernelUnit &unit) {
  if (unit.kind() == KernelUnitKind::HandwrittenException) {
    return "registered_opencl_tile_handwritten_exception";
  }
  return "registered_opencl_tile_kernel_unit";
}

std::shared_ptr<const ::ov::gfx_plugin::KernelArtifactPayload>
materialize_tile_payload(KernelArtifactDescriptor &descriptor,
                         GfxOpenClSourceArtifact artifact) {
  if (!artifact.valid || !artifact.artifact_ref.valid ||
      artifact.artifact_ref.kind != GfxKernelArtifactKind::OpenClSource ||
      artifact.artifact_ref.backend_domain != GfxKernelBackendDomain::OpenCl ||
      descriptor.kernel.backend_domain != "opencl" ||
      descriptor.payload_kind != KernelArtifactPayloadKind::OpenClSource ||
      descriptor.kernel.origin != KernelArtifactOrigin::Generated ||
      descriptor.kernel.kernel_id != artifact.artifact_ref.source_id) {
    return {};
  }

  descriptor.entry_point = artifact.artifact_ref.entry_point;
  descriptor.compile_options_key =
      gfx_opencl_source_artifact_build_options(artifact);
  descriptor.abi_arg_count = artifact.arg_count;
  descriptor.abi_output_arg_count = artifact.direct_output_count;
  return std::make_shared<GfxOpenClSourceArtifactPayload>(std::move(artifact));
}

} // namespace

KernelUnit
resolve_opencl_tile_kernel_unit(const std::shared_ptr<const ov::Node> &node,
                                const KernelRegistry &registry) {
  const auto artifact = make_opencl_tile_source_artifact(node);
  if (!artifact || !artifact->valid) {
    return {};
  }
  return registry.resolve(LoweringRouteKind::GeneratedKernel,
                          artifact->artifact_ref.source_id);
}

OperationSupportResult
query_opencl_tile_operation(const std::shared_ptr<const ov::Node> &node,
                            const KernelRegistry &registry) {
  if (!is_opencl_tile_node(node)) {
    return make_unsupported_operation("not_opencl_tile");
  }
  const auto unit = resolve_opencl_tile_kernel_unit(node, registry);
  if (!unit.valid()) {
    return make_unsupported_operation("missing_opencl_tile_kernel_unit");
  }
  return make_supported_operation(support_reason_for_tile_unit(unit),
                                  unit.route_kind(), 0.55, unit.id());
}

std::shared_ptr<const ::ov::gfx_plugin::KernelArtifactPayload>
build_opencl_tile_kernel_artifact_payload(KernelArtifactDescriptor &descriptor,
                                          const PlannedOperation &op) {
  if (descriptor.kernel.backend_domain != "opencl" ||
      descriptor.payload_kind != KernelArtifactPayloadKind::OpenClSource ||
      !op.source_node || !is_opencl_tile_node(op.source_node)) {
    return {};
  }

  auto artifact = make_opencl_tile_source_artifact(
      op.source_node, descriptor.kernel.kernel_id);
  if (!artifact || !artifact->valid) {
    return {};
  }
  return materialize_tile_payload(descriptor, std::move(*artifact));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
