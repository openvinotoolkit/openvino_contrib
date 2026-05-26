// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_operation_support.hpp"

#include <string_view>

#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "mlir/mlir_support.hpp"
#include "openvino/op/interpolate.hpp"

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
    if (mlir_supports_node(node)) {
        return make_supported_operation("generated_kernel",
                                        LoweringRouteKind::GeneratedKernel,
                                        0.5);
    }
    return make_unsupported_operation("unsupported_by_opencl_capabilities");
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
