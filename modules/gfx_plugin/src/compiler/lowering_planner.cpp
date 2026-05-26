// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/lowering_planner.hpp"

#include <sstream>
#include <utility>

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

namespace {

std::string shape_to_string(const ov::PartialShape& shape) {
    std::ostringstream os;
    os << shape;
    return os.str();
}

std::vector<std::string> input_element_types(const std::shared_ptr<const ov::Node>& node) {
    std::vector<std::string> types;
    types.reserve(node->get_input_size());
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        types.push_back(node->get_input_element_type(i).get_type_name());
    }
    return types;
}

std::vector<std::string> input_shapes(const std::shared_ptr<const ov::Node>& node) {
    std::vector<std::string> shapes;
    shapes.reserve(node->get_input_size());
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        shapes.push_back(shape_to_string(node->get_input_partial_shape(i)));
    }
    return shapes;
}

std::vector<std::string> output_element_types(const std::shared_ptr<const ov::Node>& node) {
    std::vector<std::string> types;
    types.reserve(node->get_output_size());
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        types.push_back(node->get_output_element_type(i).get_type_name());
    }
    return types;
}

std::vector<std::string> output_shapes(const std::shared_ptr<const ov::Node>& node) {
    std::vector<std::string> shapes;
    shapes.reserve(node->get_output_size());
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        shapes.push_back(shape_to_string(node->get_output_partial_shape(i)));
    }
    return shapes;
}

KernelUnitKind kind_for_dynamic_route(LoweringRouteKind route_kind) {
    switch (route_kind) {
        case LoweringRouteKind::Common:
            return KernelUnitKind::Common;
        case LoweringRouteKind::Metadata:
            return KernelUnitKind::Metadata;
        case LoweringRouteKind::VendorPrimitive:
            return KernelUnitKind::VendorPrimitive;
        case LoweringRouteKind::GeneratedKernel:
            return KernelUnitKind::GeneratedKernel;
        case LoweringRouteKind::HandwrittenKernelException:
            return KernelUnitKind::HandwrittenException;
        case LoweringRouteKind::BackendLowering:
        case LoweringRouteKind::Unsupported:
        default:
            return KernelUnitKind::BackendLowering;
    }
}

}  // namespace

bool LoweringPlan::executable() const {
    if (!unsupported.type_counts.empty()) {
        return false;
    }
    for (const auto& op : operations) {
        if (!op.kernel_unit.valid()) {
            return false;
        }
    }
    return true;
}

size_t LoweringPlan::route_count(LoweringRouteKind route_kind) const {
    size_t count = 0;
    for (const auto& op : operations) {
        if (op.kernel_unit.route_kind() == route_kind) {
            ++count;
        }
    }
    return count;
}

LoweringPlanner::LoweringPlanner(BackendTarget target)
    : LoweringPlanner(target, make_common_kernel_registry(target)) {}

LoweringPlanner::LoweringPlanner(BackendTarget target, KernelRegistry kernel_registry)
    : m_target(std::move(target)),
      m_kernel_registry(std::move(kernel_registry)) {}

LoweringPlan LoweringPlanner::plan(const std::shared_ptr<const ov::Model>& model,
                                   const OperationLegalizer& legalizer) const {
    OPENVINO_ASSERT(model, "GFX: model is null");

    const auto legality = legalizer.legalize_model(model);

    LoweringPlan plan;
    plan.target = m_target;
    plan.unsupported = legality.unsupported_summary();
    plan.operations.reserve(legality.records.size());
    for (const auto& record : legality.records) {
        if (!record.node || !record.support.semantic_legal) {
            continue;
        }
        auto kernel_unit = m_kernel_registry.resolve(record.support.preferred_route_kind,
                                                     record.support.preferred_route);
        if (!kernel_unit.valid() && !record.support.preferred_route.empty()) {
            kernel_unit = KernelUnit::describe(record.support.preferred_route_kind,
                                               kind_for_dynamic_route(record.support.preferred_route_kind),
                                               record.support.preferred_route,
                                               m_target.backend_id(),
                                               record.node->get_type_name());
        }
        plan.operations.push_back({record.node,
                                   record.node->get_friendly_name(),
                                   record.node->get_type_name(),
                                   kernel_unit,
                                   record.support.profitability_score,
                                   input_element_types(record.node),
                                   input_shapes(record.node),
                                   output_element_types(record.node),
                                   output_shapes(record.node)});
    }
    return plan;
}

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
