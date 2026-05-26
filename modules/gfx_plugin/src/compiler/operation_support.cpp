// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/operation_support.hpp"

#include <utility>

#include "compiler/backend_registry.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/assign_base.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_stage_policy.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace ov {
namespace gfx_plugin {

namespace compiler {
namespace {

bool is_view_node(const std::shared_ptr<const ov::Node>& node) {
    if (!node) {
        return false;
    }
    return select_tensor_layout_plan(node->get_type_name(), node).view_only;
}

bool is_decompression_node(const std::shared_ptr<const ov::Node>& node) {
    return node && ov::is_decompression(std::const_pointer_cast<ov::Node>(node));
}

OperationSupportResult query_common_operation(const std::shared_ptr<const ov::Node>& node) {
    if (ov::as_type_ptr<const ov::op::v0::Parameter>(node) ||
        ov::as_type_ptr<const ov::op::v0::Constant>(node) ||
        ov::as_type_ptr<const ov::op::v0::Result>(node) ||
        ov::as_type_ptr<const ov::op::util::AssignBase>(node)) {
        return make_supported_operation("common_io", LoweringRouteKind::Common, 1.0);
    }
    if (is_decompression_node(node)) {
        return make_supported_operation("decompression_metadata", LoweringRouteKind::Metadata, 1.0);
    }
    if (is_view_node(node)) {
        return make_supported_operation("view_only", LoweringRouteKind::Metadata, 1.0);
    }
    return {};
}

void log_unsupported_operation(const BackendTarget& target,
                               const std::shared_ptr<const ov::Node>& node,
                               const OperationSupportResult& result) {
    if (!result.semantic_legal && node && gfx_log_debug_enabled()) {
        gfx_log_debug("Compiler") << "Unsupported node for " << target.debug_string()
                                  << ": " << node->get_friendly_name()
                                  << " (" << node->get_type_name() << ")"
                                  << " reason=" << result.semantic_reason;
    }
}

}  // namespace

std::string_view lowering_route_kind_to_string(LoweringRouteKind kind) noexcept {
    switch (kind) {
        case LoweringRouteKind::Unsupported:
            return "unsupported";
        case LoweringRouteKind::Common:
            return "common";
        case LoweringRouteKind::Metadata:
            return "metadata";
        case LoweringRouteKind::VendorPrimitive:
            return "vendor_primitive";
        case LoweringRouteKind::GeneratedKernel:
            return "generated_kernel";
        case LoweringRouteKind::HandwrittenKernelException:
            return "handwritten_kernel_exception";
        case LoweringRouteKind::BackendLowering:
            return "backend_lowering";
    }
    return "unsupported";
}

OperationSupportResult make_supported_operation(std::string semantic_reason,
                                                LoweringRouteKind route_kind,
                                                double profitability_score,
                                                std::string preferred_route) {
    OperationSupportResult result;
    result.semantic_legal = true;
    result.semantic_reason = std::move(semantic_reason);
    result.preferred_route = std::move(preferred_route);
    result.preferred_route_kind = route_kind;
    result.profitability_score = profitability_score;
    return result;
}

OperationSupportResult make_unsupported_operation(std::string semantic_reason) {
    OperationSupportResult result;
    result.semantic_reason = std::move(semantic_reason);
    return result;
}

BackendCapabilities::BackendCapabilities(BackendTarget target,
                                         std::shared_ptr<const OperationSupportPolicy> operation_policy)
    : m_target(std::move(target)),
      m_operation_policy(std::move(operation_policy)) {}

OperationSupportResult BackendCapabilities::query_operation(const OperationSupportQuery& query) const {
    if (!query.node) {
        return make_unsupported_operation("null_node");
    }
    auto common = query_common_operation(query.node);
    if (common.semantic_legal) {
        return common;
    }

    if (!m_operation_policy) {
        return make_unsupported_operation("missing_operation_policy");
    }
    auto result = m_operation_policy->query_operation(query);
    log_unsupported_operation(m_target, query.node, result);
    return result;
}

bool BackendCapabilities::supports_node(const std::shared_ptr<const ov::Node>& node) const {
    return query_operation({node}).semantic_legal;
}

bool is_supported_node(const std::shared_ptr<const ov::Node>& node, GpuBackend backend) {
    const auto module = BackendRegistry::default_registry().resolve(backend);
    return module && module->capabilities().supports_node(node);
}

bool model_supported_by_backend(const std::shared_ptr<const ov::Model>& model, GpuBackend backend) {
    const auto module = BackendRegistry::default_registry().resolve(backend);
    return module && module->legalizer().legalize_model(model).semantic_legal();
}

UnsupportedSummary collect_unsupported(const std::shared_ptr<const ov::Model>& model,
                                       GpuBackend backend,
                                       size_t max_nodes) {
    const auto module = BackendRegistry::default_registry().resolve(backend);
    if (!module) {
        UnsupportedSummary summary;
        summary.type_counts.emplace_back("UnregisteredBackend", 1);
        summary.node_names.emplace_back(std::string("backend=") + backend_to_string(backend));
        return summary;
    }
    return module->legalizer().legalize_model(model).unsupported_summary(max_nodes);
}

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
