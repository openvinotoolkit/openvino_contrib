// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include "kernel_ir/gfx_runtime_shape_rule.hpp"
#include "runtime/gfx_stage_runtime_values.hpp"

namespace ov {
namespace gfx_plugin {

struct InferStage;

struct RuntimeShapeMaterializationRule {
    RuntimeShapeRuleKind kind = RuntimeShapeRuleKind::Unsupported;
    std::string_view runtime_shape_rule;
    bool requires_descriptor = true;
};

struct RuntimeShapeMaterializationRequest {
    RuntimeInputResolver inputs;
    const RuntimeStageExecutableDescriptor* descriptor = nullptr;
    std::vector<GpuTensor*> outputs;
    std::string_view stage_name;
    const char* error_prefix = "GFX";
};

struct RuntimeShapeMaterializationResult {
    RuntimeShapeRuleKind kind = RuntimeShapeRuleKind::Unsupported;
    RuntimeValuePlan values;
    RuntimeSelectPlan select;
    RuntimeTilePlan tile;
    RuntimeConcatPlan concat;
    RuntimeSlicePlan slice;
    bool materialized = false;
};

std::optional<RuntimeShapeMaterializationRule>
runtime_shape_materialization_rule_for(
    std::string_view runtime_shape_rule) noexcept;

bool runtime_shape_materialization_rule_supported(
    std::string_view runtime_shape_rule) noexcept;

RuntimeShapeMaterializationResult materialize_runtime_output_shapes(
    const RuntimeShapeMaterializationRequest& request);

void materialize_runtime_stage_output_shapes(InferStage& stage,
                                             const std::vector<GpuTensor*>& inputs,
                                             const char* error_prefix);

}  // namespace gfx_plugin
}  // namespace ov
