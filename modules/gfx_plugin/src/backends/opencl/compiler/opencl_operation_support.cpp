// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_operation_support.hpp"

#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "mlir/mlir_support.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

OperationSupportResult query_opencl_operation(const std::shared_ptr<const ov::Node>& node) {
    if (auto artifact = resolve_gfx_opencl_source_artifact(node)) {
        return make_supported_operation("handwritten_kernel_exception",
                                        LoweringRouteKind::HandwrittenKernelException,
                                        0.25,
                                        artifact->artifact_ref.source_id);
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
