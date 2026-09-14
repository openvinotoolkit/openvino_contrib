// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_eltwise_kernel_unit.hpp"

#include <string>
#include <utility>

#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/lowering_planner.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

std::string support_reason_for_eltwise_unit(const KernelUnit &unit) {
  if (unit.kind() == KernelUnitKind::HandwrittenException) {
    return "registered_opencl_eltwise_handwritten_exception";
  }
  return "registered_opencl_eltwise_kernel_unit";
}

std::shared_ptr<const ::ov::gfx_plugin::KernelArtifactPayload>
materialize_eltwise_payload(const KernelArtifactDescriptor &descriptor,
                            GfxOpenClSourceArtifact artifact) {
  if (!opencl_source_artifact_matches_descriptor_contract(descriptor,
                                                          artifact)) {
    return {};
  }
  return std::make_shared<GfxOpenClSourceArtifactPayload>(std::move(artifact));
}

} // namespace

bool is_opencl_eltwise_node(const std::shared_ptr<const ov::Node> &node) {
  return ov::as_type_ptr<const ov::op::util::BinaryElementwiseArithmetic>(
             node) ||
         ov::as_type_ptr<const ov::op::util::BinaryElementwiseComparison>(
             node) ||
         ov::as_type_ptr<const ov::op::util::BinaryElementwiseLogical>(node) ||
         ov::as_type_ptr<const ov::op::v1::LogicalNot>(node) ||
         ov::as_type_ptr<const ov::op::v1::Select>(node);
}

KernelUnit
resolve_opencl_eltwise_kernel_unit(const std::shared_ptr<const ov::Node> &node,
                                   const KernelRegistry &registry) {
  const auto artifact = make_opencl_eltwise_source_artifact(node);
  if (!artifact || !artifact->valid) {
    return {};
  }
  return registry.resolve(LoweringRouteKind::GeneratedKernel,
                          artifact->artifact_ref.source_id);
}

OperationSupportResult
query_opencl_eltwise_operation(const std::shared_ptr<const ov::Node> &node,
                               const KernelRegistry &registry) {
  if (!is_opencl_eltwise_node(node)) {
    return make_unsupported_operation("not_opencl_eltwise");
  }
  const auto unit = resolve_opencl_eltwise_kernel_unit(node, registry);
  if (!unit.valid()) {
    return make_unsupported_operation("missing_opencl_eltwise_kernel_unit");
  }
  return make_supported_operation(support_reason_for_eltwise_unit(unit),
                                  unit.route_kind(), 0.55, unit.id());
}

std::shared_ptr<const ::ov::gfx_plugin::KernelArtifactPayload>
build_opencl_eltwise_kernel_artifact_payload(
    const KernelArtifactDescriptor &descriptor, const PlannedOperation &op) {
  if (descriptor.kernel.backend_domain != "opencl" ||
      descriptor.payload_kind != KernelArtifactPayloadKind::OpenClSource ||
      !op.source_node || !is_opencl_eltwise_node(op.source_node)) {
    return {};
  }

  auto artifact = make_opencl_eltwise_source_artifact(
      op.source_node, descriptor.kernel.kernel_id);
  if (!artifact || !artifact->valid) {
    return {};
  }
  return materialize_eltwise_payload(descriptor, std::move(*artifact));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
