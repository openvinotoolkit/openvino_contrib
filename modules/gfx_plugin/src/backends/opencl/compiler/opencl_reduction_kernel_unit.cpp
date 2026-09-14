// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_reduction_kernel_unit.hpp"

#include <string>
#include <utility>

#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/lowering_planner.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

std::string support_reason_for_reduction_unit(const KernelUnit &unit) {
  if (unit.kind() == KernelUnitKind::HandwrittenException) {
    return "registered_opencl_reduction_handwritten_exception";
  }
  return "registered_opencl_reduction_kernel_unit";
}

std::shared_ptr<const ::ov::gfx_plugin::KernelArtifactPayload>
materialize_reduction_payload(const KernelArtifactDescriptor &descriptor,
                              GfxOpenClSourceArtifact artifact) {
  if (!opencl_source_artifact_matches_descriptor_contract(descriptor,
                                                          artifact)) {
    return {};
  }
  return std::make_shared<GfxOpenClSourceArtifactPayload>(std::move(artifact));
}

} // namespace

bool is_opencl_reduction_node(const std::shared_ptr<const ov::Node> &node) {
  return ov::as_type_ptr<const ov::op::v1::ReduceSum>(node) ||
         ov::as_type_ptr<const ov::op::v1::ReduceMean>(node) ||
         ov::as_type_ptr<const ov::op::v1::ReduceMax>(node) ||
         ov::as_type_ptr<const ov::op::v1::ReduceMin>(node) ||
         ov::as_type_ptr<const ov::op::v1::ReduceProd>(node) ||
         ov::as_type_ptr<const ov::op::v4::ReduceL1>(node) ||
         ov::as_type_ptr<const ov::op::v4::ReduceL2>(node) ||
         ov::as_type_ptr<const ov::op::v1::ReduceLogicalAnd>(node) ||
         ov::as_type_ptr<const ov::op::v1::ReduceLogicalOr>(node);
}

KernelUnit resolve_opencl_reduction_kernel_unit(
    const std::shared_ptr<const ov::Node> &node,
    const KernelRegistry &registry) {
  const auto artifact = make_opencl_reduction_source_artifact(node);
  if (!artifact || !artifact->valid) {
    return {};
  }
  return registry.resolve(LoweringRouteKind::GeneratedKernel,
                          artifact->artifact_ref.source_id);
}

OperationSupportResult
query_opencl_reduction_operation(const std::shared_ptr<const ov::Node> &node,
                                 const KernelRegistry &registry) {
  if (!is_opencl_reduction_node(node)) {
    return make_unsupported_operation("not_opencl_reduction");
  }
  const auto unit = resolve_opencl_reduction_kernel_unit(node, registry);
  if (!unit.valid()) {
    return make_unsupported_operation("missing_opencl_reduction_kernel_unit");
  }
  return make_supported_operation(support_reason_for_reduction_unit(unit),
                                  unit.route_kind(), 0.55, unit.id());
}

std::shared_ptr<const ::ov::gfx_plugin::KernelArtifactPayload>
build_opencl_reduction_kernel_artifact_payload(
    const KernelArtifactDescriptor &descriptor, const PlannedOperation &op) {
  if (descriptor.kernel.backend_domain != "opencl" ||
      descriptor.payload_kind != KernelArtifactPayloadKind::OpenClSource ||
      !op.source_node || !is_opencl_reduction_node(op.source_node)) {
    return {};
  }

  auto artifact = make_opencl_reduction_source_artifact(
      op.source_node, descriptor.kernel.kernel_id);
  if (!artifact || !artifact->valid) {
    return {};
  }
  return materialize_reduction_payload(descriptor, std::move(*artifact));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
