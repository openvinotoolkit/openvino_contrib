// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <string_view>

#include "compiler/operation_support.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

enum class KernelUnitKind {
    Common,
    Metadata,
    VendorPrimitive,
    GeneratedKernel,
    HandwrittenException,
    BackendLowering,
};

class KernelUnit final {
public:
    KernelUnit() = default;

    static KernelUnit from_route(LoweringRouteKind route_kind, std::string unit_id);
    static KernelUnit describe(LoweringRouteKind route_kind,
                               KernelUnitKind kind,
                               std::string unit_id,
                               std::string backend_domain,
                               std::string op_family);

    const std::string& id() const noexcept {
        return m_id;
    }

    KernelUnitKind kind() const noexcept {
        return m_kind;
    }

    LoweringRouteKind route_kind() const noexcept {
        return m_route_kind;
    }

    const std::string& backend_domain() const noexcept {
        return m_backend_domain;
    }

    const std::string& op_family() const noexcept {
        return m_op_family;
    }

    bool valid() const noexcept {
        return m_route_kind != LoweringRouteKind::Unsupported && !m_id.empty();
    }

    std::string manifest_key() const;

private:
    KernelUnit(LoweringRouteKind route_kind,
               KernelUnitKind kind,
               std::string unit_id,
               std::string backend_domain,
               std::string op_family);

    KernelUnitKind m_kind = KernelUnitKind::BackendLowering;
    LoweringRouteKind m_route_kind = LoweringRouteKind::Unsupported;
    std::string m_id;
    std::string m_backend_domain;
    std::string m_op_family;
};

std::string_view kernel_unit_kind_to_string(KernelUnitKind kind) noexcept;
std::string_view kernel_unit_route_to_string(const KernelUnit& unit) noexcept;

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
