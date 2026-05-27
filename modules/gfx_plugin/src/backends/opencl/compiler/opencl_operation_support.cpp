// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_operation_support.hpp"

#include <string_view>

#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

bool is_generated_opencl_kernel_unit(const GfxOpenClSourceArtifact& artifact) {
    constexpr std::string_view kGeneratedPrefix = "opencl/generated/";
    const std::string_view source_id = artifact.artifact_ref.source_id;
    return source_id.size() >= kGeneratedPrefix.size() &&
           source_id.substr(0, kGeneratedPrefix.size()) == kGeneratedPrefix;
}

bool is_interpolate_node(const std::shared_ptr<const ov::Node>& node) {
    return ov::as_type_ptr<const ov::op::v0::Interpolate>(node) ||
           ov::as_type_ptr<const ov::op::v4::Interpolate>(node) ||
           ov::as_type_ptr<const ov::op::v11::Interpolate>(node);
}

bool is_matmul_node(const std::shared_ptr<const ov::Node>& node) {
    return static_cast<bool>(ov::as_type_ptr<const ov::op::v0::MatMul>(node));
}

bool is_activation_node(const std::shared_ptr<const ov::Node>& node) {
    return static_cast<bool>(
        ov::as_type_ptr<const ov::op::util::UnaryElementwiseArithmetic>(node));
}

bool is_eltwise_node(const std::shared_ptr<const ov::Node>& node) {
    return ov::as_type_ptr<const ov::op::util::BinaryElementwiseArithmetic>(node) ||
           ov::as_type_ptr<const ov::op::util::BinaryElementwiseComparison>(node) ||
           ov::as_type_ptr<const ov::op::util::BinaryElementwiseLogical>(node);
}

OperationSupportResult query_opencl_operation(const std::shared_ptr<const ov::Node>& node) {
    if (auto artifact = resolve_gfx_opencl_source_artifact(node)) {
        const bool generated = is_generated_opencl_kernel_unit(*artifact);
        return make_supported_operation(generated ? "generated_opencl_kernel_unit"
                                                  : "handwritten_kernel_exception",
                                        generated ? LoweringRouteKind::GeneratedKernel
                                                  : LoweringRouteKind::HandwrittenKernelException,
                                        generated ? 0.55 : 0.25,
                                        artifact->artifact_ref.source_id);
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
    return make_unsupported_operation("missing_opencl_kernel_unit");
}

class OpenCLOperationSupportPolicy final : public OperationSupportPolicy {
public:
    OperationSupportResult query_operation(const OperationSupportQuery& query) const override {
        return query_opencl_operation(query.node);
    }
};

}  // namespace

std::shared_ptr<const OperationSupportPolicy> make_opencl_operation_support_policy() {
    static const auto policy = std::make_shared<OpenCLOperationSupportPolicy>();
    return policy;
}

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
