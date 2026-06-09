// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_activation_kernel_unit.hpp"

#include <string>
#include <utility>

#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/lowering_planner.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "kernel_ir/opencl_kernels/activation_kernel.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

std::string support_reason_for_activation_unit(const KernelUnit &unit) {
  if (unit.kind() == KernelUnitKind::HandwrittenException) {
    return "registered_opencl_activation_handwritten_exception";
  }
  return "registered_opencl_activation_kernel_unit";
}

std::shared_ptr<const ::ov::gfx_plugin::KernelArtifactPayload>
materialize_activation_payload(const KernelArtifactDescriptor &descriptor,
                               GfxOpenClSourceArtifact artifact) {
  if (!opencl_source_artifact_matches_descriptor_contract(descriptor,
                                                          artifact)) {
    return {};
  }
  return std::make_shared<GfxOpenClSourceArtifactPayload>(std::move(artifact));
}

} // namespace

bool is_opencl_activation_node(const std::shared_ptr<const ov::Node> &node) {
  return ov::as_type_ptr<const ov::op::util::UnaryElementwiseArithmetic>(
             node) ||
         ov::as_type_ptr<const ov::op::v4::Swish>(node);
}

KernelUnit resolve_opencl_activation_kernel_unit(
    const std::shared_ptr<const ov::Node> &node,
    const KernelRegistry &registry) {
  const auto artifact = make_opencl_activation_source_artifact(node);
  if (!artifact || !artifact->valid) {
    return {};
  }
  return registry.resolve(LoweringRouteKind::GeneratedKernel,
                          artifact->artifact_ref.source_id);
}

OperationSupportResult
query_opencl_activation_operation(const std::shared_ptr<const ov::Node> &node,
                                  const KernelRegistry &registry) {
  if (!is_opencl_activation_node(node)) {
    return make_unsupported_operation("not_opencl_activation");
  }
  const auto unit = resolve_opencl_activation_kernel_unit(node, registry);
  if (!unit.valid()) {
    return make_unsupported_operation("missing_opencl_activation_kernel_unit");
  }
  return make_supported_operation(support_reason_for_activation_unit(unit),
                                  unit.route_kind(), 0.55, unit.id());
}

std::shared_ptr<const ::ov::gfx_plugin::KernelArtifactPayload>
build_opencl_activation_kernel_artifact_payload(
    const KernelArtifactDescriptor &descriptor, const PlannedOperation &op) {
  if (descriptor.kernel.backend_domain != "opencl" ||
      descriptor.payload_kind != KernelArtifactPayloadKind::OpenClSource ||
      !op.source_node || !is_opencl_activation_node(op.source_node)) {
    return {};
  }

  auto artifact = make_opencl_activation_source_artifact(
      op.source_node, descriptor.kernel.kernel_id);
  if (!artifact || !artifact->valid) {
    return {};
  }
  return materialize_activation_payload(descriptor, std::move(*artifact));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
