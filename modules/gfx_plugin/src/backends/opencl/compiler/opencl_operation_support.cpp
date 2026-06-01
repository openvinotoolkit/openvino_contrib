// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_operation_support.hpp"

#include <string_view>

#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

bool is_generated_opencl_kernel_unit(const GfxOpenClSourceArtifact &artifact) {
  constexpr std::string_view kGeneratedPrefix = "opencl/generated/";
  const std::string_view source_id = artifact.artifact_ref.source_id;
  return source_id.size() >= kGeneratedPrefix.size() &&
         source_id.substr(0, kGeneratedPrefix.size()) == kGeneratedPrefix;
}

bool is_interpolate_node(const std::shared_ptr<const ov::Node> &node) {
  return ov::as_type_ptr<const ov::op::v0::Interpolate>(node) ||
         ov::as_type_ptr<const ov::op::v4::Interpolate>(node) ||
         ov::as_type_ptr<const ov::op::v11::Interpolate>(node);
}

bool is_matmul_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(ov::as_type_ptr<const ov::op::v0::MatMul>(node));
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

bool is_pooling_node(const std::shared_ptr<const ov::Node> &node) {
  return ov::as_type_ptr<const ov::op::util::MaxPoolBase>(node) ||
         ov::as_type_ptr<const ov::op::util::AvgPoolBase>(node);
}

bool is_softmax_node(const std::shared_ptr<const ov::Node> &node) {
  return ov::as_type_ptr<const ov::op::v1::Softmax>(node) ||
         ov::as_type_ptr<const ov::op::v8::Softmax>(node);
}

bool is_transpose_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Transpose>(node));
}

OperationSupportResult
query_opencl_operation(const std::shared_ptr<const ov::Node> &node) {
  if (auto artifact = resolve_gfx_opencl_source_artifact(node)) {
    const bool generated = is_generated_opencl_kernel_unit(*artifact);
    return make_supported_operation(
        generated ? "generated_opencl_kernel_unit"
                  : "handwritten_kernel_exception",
        generated ? LoweringRouteKind::GeneratedKernel
                  : LoweringRouteKind::HandwrittenKernelException,
        generated ? 0.55 : 0.25, artifact->artifact_ref.source_id);
  }
  if (is_interpolate_node(node)) {
    return make_unsupported_operation("missing_opencl_interpolate_kernel_unit");
  }
  if (is_matmul_node(node)) {
    return make_unsupported_operation("missing_opencl_matmul_kernel_unit");
  }
  if (is_activation_node(node)) {
    return make_unsupported_operation("missing_opencl_activation_kernel_unit");
  }
  if (is_eltwise_node(node)) {
    return make_unsupported_operation("missing_opencl_eltwise_kernel_unit");
  }
  if (is_reduction_node(node)) {
    return make_unsupported_operation("missing_opencl_reduction_kernel_unit");
  }
  if (is_pooling_node(node)) {
    return make_unsupported_operation("missing_opencl_pooling_kernel_unit");
  }
  if (is_softmax_node(node)) {
    return make_unsupported_operation("missing_opencl_softmax_kernel_unit");
  }
  if (is_transpose_node(node)) {
    return make_unsupported_operation("missing_opencl_transpose_kernel_unit");
  }
  return make_unsupported_operation("missing_opencl_kernel_unit");
}

class OpenCLOperationSupportPolicy final : public OperationSupportPolicy {
public:
  OperationSupportResult
  query_operation(const OperationSupportQuery &query) const override {
    return query_opencl_operation(query.node);
  }
};

} // namespace

std::shared_ptr<const OperationSupportPolicy>
make_opencl_operation_support_policy() {
  static const auto policy = std::make_shared<OpenCLOperationSupportPolicy>();
  return policy;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
