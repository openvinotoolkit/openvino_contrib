// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_operation_support.hpp"

#include <utility>

#include "backends/opencl/compiler/opencl_conv_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "backends/opencl/compiler/opencl_pool_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_range_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_softmax_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_tile_kernel_unit.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

KernelUnit
resolve_opencl_source_kernel_unit(const GfxOpenClSourceArtifact &artifact,
                                  const KernelRegistry &kernel_registry) {
  switch (
      classify_opencl_kernel_artifact_origin(artifact.artifact_ref.source_id)) {
  case KernelArtifactOrigin::Generated:
    return kernel_registry.resolve(LoweringRouteKind::GeneratedKernel,
                                   artifact.artifact_ref.source_id);
  case KernelArtifactOrigin::HandwrittenException:
    return kernel_registry.resolve(
        LoweringRouteKind::HandwrittenKernelException,
        artifact.artifact_ref.source_id);
  case KernelArtifactOrigin::Common:
  case KernelArtifactOrigin::Metadata:
  case KernelArtifactOrigin::VendorPrimitive:
  case KernelArtifactOrigin::Unknown:
  default:
    return {};
  }
}

OperationSupportResult
query_explicit_opencl_source_family(const std::shared_ptr<const ov::Node> &node,
                                    const KernelRegistry &kernel_registry,
                                    const char *support_reason,
                                    const char *missing_reason) {
  const auto artifact = resolve_gfx_opencl_source_artifact(node);
  if (!artifact || !artifact->valid) {
    return make_unsupported_operation(missing_reason);
  }
  if (!is_explicit_opencl_source_artifact_unit(
          artifact->artifact_ref.source_id) ||
      classify_opencl_kernel_artifact_origin(
          artifact->artifact_ref.source_id) !=
          KernelArtifactOrigin::Generated) {
    return make_unsupported_operation(missing_reason);
  }
  const auto unit =
      resolve_opencl_source_kernel_unit(*artifact, kernel_registry);
  if (!unit.valid()) {
    return make_unsupported_operation(missing_reason);
  }
  return make_supported_operation(
      support_reason, unit.route_kind(),
      unit.kind() == KernelUnitKind::HandwrittenException ? 0.25 : 0.55,
      unit.id());
}

bool is_interpolate_node(const std::shared_ptr<const ov::Node> &node) {
  return ov::as_type_ptr<const ov::op::v0::Interpolate>(node) ||
         ov::as_type_ptr<const ov::op::v4::Interpolate>(node) ||
         ov::as_type_ptr<const ov::op::v11::Interpolate>(node);
}

bool is_matmul_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(ov::as_type_ptr<const ov::op::v0::MatMul>(node));
}

bool is_convolution_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(
      ov::as_type_ptr<const ov::op::v1::Convolution>(node));
}

bool is_group_convolution_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(
      ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node));
}

bool is_activation_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(
      ov::as_type_ptr<const ov::op::util::UnaryElementwiseArithmetic>(node) ||
      ov::as_type_ptr<const ov::op::v4::Swish>(node));
}

bool is_eltwise_node(const std::shared_ptr<const ov::Node> &node) {
  return ov::as_type_ptr<const ov::op::util::BinaryElementwiseArithmetic>(
             node) ||
         ov::as_type_ptr<const ov::op::util::BinaryElementwiseComparison>(
             node) ||
         ov::as_type_ptr<const ov::op::util::BinaryElementwiseLogical>(node);
}

bool is_reduction_node(const std::shared_ptr<const ov::Node> &node) {
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

bool is_transpose_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Transpose>(node));
}

bool is_shapeof_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(
      ov::as_type_ptr<const ov::op::util::ShapeOfBase>(node));
}

bool is_concat_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(ov::as_type_ptr<const ov::op::v0::Concat>(node));
}

bool is_split_node(const std::shared_ptr<const ov::Node> &node) {
  return ov::as_type_ptr<const ov::op::v1::Split>(node) ||
         ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node);
}

OperationSupportResult
query_opencl_operation(const std::shared_ptr<const ov::Node> &node,
                       const KernelRegistry &kernel_registry) {
  if (is_opencl_range_node(node)) {
    return query_opencl_range_operation(node, kernel_registry);
  }
  if (is_opencl_tile_node(node)) {
    return query_opencl_tile_operation(node, kernel_registry);
  }
  if (is_opencl_softmax_node(node)) {
    return query_opencl_softmax_operation(node, kernel_registry);
  }
  if (is_convolution_node(node) || is_group_convolution_node(node)) {
    return query_opencl_conv2d_operation(node, kernel_registry);
  }
  if (is_interpolate_node(node)) {
    return query_explicit_opencl_source_family(
        node, kernel_registry, "registered_opencl_interpolate_kernel_unit",
        "missing_opencl_interpolate_kernel_unit");
  }
  if (is_matmul_node(node)) {
    return query_explicit_opencl_source_family(
        node, kernel_registry, "registered_opencl_matmul_kernel_unit",
        "missing_opencl_matmul_kernel_unit");
  }
  if (is_activation_node(node)) {
    return query_explicit_opencl_source_family(
        node, kernel_registry, "registered_opencl_activation_kernel_unit",
        "missing_opencl_activation_kernel_unit");
  }
  if (is_eltwise_node(node)) {
    return query_explicit_opencl_source_family(
        node, kernel_registry, "registered_opencl_eltwise_kernel_unit",
        "missing_opencl_eltwise_kernel_unit");
  }
  if (is_reduction_node(node)) {
    return query_explicit_opencl_source_family(
        node, kernel_registry, "registered_opencl_reduction_kernel_unit",
        "missing_opencl_reduction_kernel_unit");
  }
  if (is_opencl_pool2d_node(node)) {
    return query_opencl_pool2d_operation(node, kernel_registry);
  }
  if (is_transpose_node(node)) {
    return query_explicit_opencl_source_family(
        node, kernel_registry, "registered_opencl_transpose_kernel_unit",
        "missing_opencl_transpose_kernel_unit");
  }
  if (is_shapeof_node(node)) {
    return query_explicit_opencl_source_family(
        node, kernel_registry, "registered_opencl_shapeof_kernel_unit",
        "missing_opencl_shapeof_kernel_unit");
  }
  if (is_concat_node(node)) {
    return query_explicit_opencl_source_family(
        node, kernel_registry, "registered_opencl_concat_kernel_unit",
        "missing_opencl_concat_kernel_unit");
  }
  if (is_split_node(node)) {
    return query_explicit_opencl_source_family(
        node, kernel_registry, "registered_opencl_split_kernel_unit",
        "missing_opencl_split_kernel_unit");
  }
  if (auto artifact = resolve_gfx_opencl_source_artifact(node);
      artifact && artifact->valid &&
      is_explicit_opencl_source_artifact_unit(
          artifact->artifact_ref.source_id)) {
    return query_explicit_opencl_source_family(
        node, kernel_registry,
        "registered_opencl_explicit_generated_source_kernel_unit",
        "missing_opencl_kernel_unit");
  }
  return make_unsupported_operation("missing_opencl_kernel_unit");
}

class OpenCLOperationSupportPolicy final : public OperationSupportPolicy {
public:
  explicit OpenCLOperationSupportPolicy(KernelRegistry kernel_registry)
      : m_kernel_registry(std::move(kernel_registry)) {}

  OperationSupportResult
  query_operation(const OperationSupportQuery &query) const override {
    return query_opencl_operation(query.node, m_kernel_registry);
  }

private:
  KernelRegistry m_kernel_registry;
};

} // namespace

std::shared_ptr<const OperationSupportPolicy>
make_opencl_operation_support_policy(KernelRegistry kernel_registry) {
  return std::make_shared<OpenCLOperationSupportPolicy>(
      std::move(kernel_registry));
}

std::shared_ptr<const OperationSupportPolicy>
make_opencl_operation_support_policy() {
  return make_opencl_operation_support_policy(make_opencl_kernel_registry(
      BackendTarget::from_backend(GpuBackend::OpenCL)));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
