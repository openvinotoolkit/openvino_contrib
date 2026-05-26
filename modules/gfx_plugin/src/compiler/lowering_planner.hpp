// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "compiler/backend_target.hpp"
#include "compiler/kernel_registry.hpp"
#include "compiler/kernel_unit.hpp"
#include "compiler/operation_legalizer.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

struct PlannedOperation {
    std::shared_ptr<const ov::Node> source_node;
    std::string node_name;
    std::string type_name;
    KernelUnit kernel_unit;
    double profitability_score = 0.0;
    std::vector<std::string> input_element_types;
    std::vector<std::string> input_shapes;
    std::vector<std::string> output_element_types;
    std::vector<std::string> output_shapes;
};

struct LoweringPlan {
    BackendTarget target = BackendTarget::from_backend(GpuBackend::Metal);
    std::vector<PlannedOperation> operations;
    UnsupportedSummary unsupported;

    bool executable() const;
    size_t route_count(LoweringRouteKind route_kind) const;
};

class LoweringPlanner final {
public:
    explicit LoweringPlanner(BackendTarget target);
    LoweringPlanner(BackendTarget target, KernelRegistry kernel_registry);

    const BackendTarget& target() const noexcept {
        return m_target;
    }

    LoweringPlan plan(const std::shared_ptr<const ov::Model>& model,
                      const OperationLegalizer& legalizer) const;

private:
    BackendTarget m_target;
    KernelRegistry m_kernel_registry;
};

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
