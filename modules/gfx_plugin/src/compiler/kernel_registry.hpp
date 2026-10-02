// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "compiler/backend_target.hpp"
#include "compiler/kernel_unit.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

struct KernelRegistryAudit {
    std::vector<std::string> diagnostics;
    size_t handwritten_exception_count = 0;

    bool valid() const noexcept {
        return diagnostics.empty();
    }
};

class KernelRegistry final {
public:
    KernelRegistry() = default;
    KernelRegistry(BackendTarget target, std::vector<KernelUnit> units);

    const BackendTarget& target() const noexcept {
        return m_target;
    }

    const std::vector<KernelUnit>& units() const noexcept {
        return m_units;
    }

    KernelUnit resolve(LoweringRouteKind route_kind, std::string_view unit_id) const;
    KernelRegistryAudit audit() const;
    size_t route_count(LoweringRouteKind route_kind) const;
    size_t handwritten_exception_count() const;

private:
    BackendTarget m_target;
    std::vector<KernelUnit> m_units;
};

std::vector<KernelUnit> make_common_kernel_units(const BackendTarget& target);
KernelRegistry make_common_kernel_registry(const BackendTarget& target);
KernelRegistry make_metal_kernel_registry(const BackendTarget& target);
KernelRegistry make_opencl_kernel_registry(const BackendTarget& target);

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
