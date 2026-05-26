// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/kernel_unit.hpp"

#include <utility>

namespace ov {
namespace gfx_plugin {
namespace compiler {

namespace {

KernelUnitKind kind_for_route(LoweringRouteKind route_kind) {
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
            return KernelUnitKind::BackendLowering;
        case LoweringRouteKind::Unsupported:
            return KernelUnitKind::BackendLowering;
    }
    return KernelUnitKind::BackendLowering;
}

std::string default_unit_id(LoweringRouteKind route_kind) {
    return std::string(lowering_route_kind_to_string(route_kind));
}

}  // namespace

KernelUnit::KernelUnit(LoweringRouteKind route_kind,
                       KernelUnitKind kind,
                       std::string unit_id,
                       std::string backend_domain,
                       std::string op_family)
    : m_kind(kind),
      m_route_kind(route_kind),
      m_id(std::move(unit_id)),
      m_backend_domain(std::move(backend_domain)),
      m_op_family(std::move(op_family)) {}

KernelUnit KernelUnit::from_route(LoweringRouteKind route_kind, std::string unit_id) {
    if (unit_id.empty() && route_kind != LoweringRouteKind::Unsupported) {
        unit_id = default_unit_id(route_kind);
    }
    return KernelUnit(route_kind, kind_for_route(route_kind), std::move(unit_id), {}, {});
}

KernelUnit KernelUnit::describe(LoweringRouteKind route_kind,
                                KernelUnitKind kind,
                                std::string unit_id,
                                std::string backend_domain,
                                std::string op_family) {
    if (unit_id.empty() && route_kind != LoweringRouteKind::Unsupported) {
        unit_id = default_unit_id(route_kind);
    }
    return KernelUnit(route_kind,
                      kind,
                      std::move(unit_id),
                      std::move(backend_domain),
                      std::move(op_family));
}

std::string KernelUnit::manifest_key() const {
    return std::string(lowering_route_kind_to_string(m_route_kind)) + "#" + m_id;
}

std::string_view kernel_unit_kind_to_string(KernelUnitKind kind) noexcept {
    switch (kind) {
        case KernelUnitKind::Common:
            return "common";
        case KernelUnitKind::Metadata:
            return "metadata";
        case KernelUnitKind::VendorPrimitive:
            return "vendor_primitive";
        case KernelUnitKind::GeneratedKernel:
            return "generated_kernel";
        case KernelUnitKind::HandwrittenException:
            return "handwritten_exception";
        case KernelUnitKind::BackendLowering:
            return "backend_lowering";
    }
    return "backend_lowering";
}

std::string_view kernel_unit_route_to_string(const KernelUnit& unit) noexcept {
    return lowering_route_kind_to_string(unit.route_kind());
}

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
