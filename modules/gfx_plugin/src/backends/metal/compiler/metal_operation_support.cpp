// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/metal_operation_support.hpp"

#include <exception>

#include "runtime/gfx_logger.hpp"

namespace ov {
namespace gfx_plugin {

bool metal_supports_node(const std::shared_ptr<const ov::Node>& node);

namespace compiler {
namespace {

OperationSupportResult query_metal_operation(const std::shared_ptr<const ov::Node>& node) {
    try {
        if (node && node->get_type_name() == std::string("ShapeOf") &&
            node->get_input_partial_shape(0).rank().is_static()) {
            return make_supported_operation("generated_msl_source",
                                            LoweringRouteKind::GeneratedKernel,
                                            0.55,
                                            "metal/generated/shapeof");
        }
        const bool supported = metal_supports_node(node);
        if (supported) {
            return make_supported_operation("backend_lowering", LoweringRouteKind::BackendLowering, 0.5);
        }
        return make_unsupported_operation("unsupported_by_metal_capabilities");
    } catch (const std::exception& e) {
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("Compiler") << "Exception probing node " << node->get_friendly_name()
                                      << " (" << node->get_type_name() << "): " << e.what();
        }
        return make_unsupported_operation(e.what());
    } catch (...) {
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("Compiler") << "Unknown exception probing node " << node->get_friendly_name()
                                      << " (" << node->get_type_name() << ")";
        }
        return make_unsupported_operation("unknown_probe_exception");
    }
}

class MetalOperationSupportPolicy final : public OperationSupportPolicy {
public:
    OperationSupportResult query_operation(const OperationSupportQuery& query) const override {
        return query_metal_operation(query.node);
    }
};

}  // namespace

std::shared_ptr<const OperationSupportPolicy> make_metal_operation_support_policy() {
    static const auto policy = std::make_shared<MetalOperationSupportPolicy>();
    return policy;
}

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
