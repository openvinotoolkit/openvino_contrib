// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/runtime_shape_materializer.hpp"

#include "runtime/gfx_stage_runtime_values.hpp"
#include "runtime/infer_pipeline.hpp"

#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace ov {
namespace gfx_plugin {

namespace {

std::string stage_runtime_shape_name(const InferStage& stage) {
    const auto* descriptor = runtime_stage_descriptor_or_null(stage);
    if (descriptor && !descriptor->stage_name.empty()) {
        return descriptor->stage_name;
    }
    return stage.stage ? stage.stage->name() : std::string("<null>");
}

std::vector<GpuTensor*> stage_output_refs(InferStage& stage) {
    std::vector<GpuTensor*> outputs;
    outputs.reserve(stage.outputs.size());
    for (auto& out : stage.outputs) {
        outputs.push_back(out.get());
    }
    return outputs;
}

struct RuntimeShapeMaterializationContext {
    RuntimeInputResolver inputs;
    const RuntimeStageExecutableDescriptor* descriptor = nullptr;
    std::vector<GpuTensor*> outputs;
    std::string_view stage_name;
    const char* error_prefix = "GFX";
    RuntimeShapeMaterializationResult result;
};

using RuntimeShapeMaterializationFn =
    void (*)(RuntimeShapeMaterializationContext&);

struct RuntimeShapeMaterializationHandler {
    RuntimeShapeRuleKind kind = RuntimeShapeRuleKind::Unsupported;
    RuntimeShapeMaterializationFn materialize = nullptr;
};

void materialize_concat(RuntimeShapeMaterializationContext& context) {
    context.result.concat =
        plan_concat_runtime_values(context.inputs, context.stage_name);
    context.result.values = context.result.concat.values;
    assign_runtime_value_outputs(context.result.values, context.outputs);
}

void materialize_broadcast(RuntimeShapeMaterializationContext& context) {
    const ov::Shape in_shape = context.inputs.shape(0);
    OPENVINO_ASSERT(!in_shape.empty(),
                    context.error_prefix,
                    ": Broadcast input shape is unknown for stage ",
                    context.stage_name);
    const auto plan = plan_broadcast_runtime_values(context.inputs,
                                                    in_shape,
                                                    context.stage_name);
    context.result.values = plan;
    assign_runtime_value_outputs(context.result.values, context.outputs);
}

void materialize_select(RuntimeShapeMaterializationContext& context) {
    context.result.select =
        plan_select_runtime_values(context.inputs, context.stage_name);
    OPENVINO_ASSERT(context.result.select.valid(),
                    context.error_prefix,
                    ": Select runtime shapes are unknown for stage ",
                    context.stage_name);
    context.result.values = context.result.select.values;
    assign_runtime_value_outputs(context.result.values, context.outputs);
}

void materialize_shape_of(RuntimeShapeMaterializationContext& context) {
    context.result.values =
        plan_shapeof_runtime_values(context.inputs, context.stage_name);
    assign_runtime_value_outputs(context.result.values, context.outputs);
}

void materialize_slice(RuntimeShapeMaterializationContext& context) {
    context.result.slice =
        plan_slice_runtime_values(context.inputs,
                                  context.outputs,
                                  context.descriptor->requires_runtime_shape_args,
                                  context.stage_name);
    context.result.values = context.result.slice.values;
    assign_runtime_value_outputs(context.result.values, context.outputs);
}

void materialize_range(RuntimeShapeMaterializationContext& context) {
    context.result.values =
        plan_range_runtime_values(context.inputs, context.stage_name);
    assign_runtime_value_outputs(context.result.values, context.outputs);
}

void materialize_tile(RuntimeShapeMaterializationContext& context) {
    context.result.tile =
        plan_tile_runtime_values(context.inputs, context.outputs, context.stage_name);
    OPENVINO_ASSERT(context.result.tile.valid(),
                    context.error_prefix,
                    ": Tile runtime shape is unknown for stage ",
                    context.stage_name);
    context.result.values = context.result.tile.values;
    assign_runtime_value_outputs(context.result.values, context.outputs);
}

constexpr std::array<RuntimeShapeMaterializationHandler, 7>
    kRuntimeShapeMaterializationHandlers = {{
        {RuntimeShapeRuleKind::Concat, &materialize_concat},
        {RuntimeShapeRuleKind::Broadcast, &materialize_broadcast},
        {RuntimeShapeRuleKind::Select, &materialize_select},
        {RuntimeShapeRuleKind::ShapeOf, &materialize_shape_of},
        {RuntimeShapeRuleKind::Slice, &materialize_slice},
        {RuntimeShapeRuleKind::Range, &materialize_range},
        {RuntimeShapeRuleKind::Tile, &materialize_tile},
    }};

const RuntimeShapeMaterializationHandler*
find_runtime_shape_materialization_handler(
    RuntimeShapeRuleKind kind) noexcept {
    for (const auto& handler : kRuntimeShapeMaterializationHandlers) {
        if (handler.kind == kind) {
            return &handler;
        }
    }
    return nullptr;
}

}  // namespace

std::optional<RuntimeShapeMaterializationRule>
runtime_shape_materialization_rule_for(
    std::string_view runtime_shape_rule) noexcept {
    const auto kind = runtime_shape_rule_kind_from_name(runtime_shape_rule);
    if (kind == RuntimeShapeRuleKind::Unsupported) {
        return std::nullopt;
    }
    if (kind == RuntimeShapeRuleKind::StaticOrDescriptor) {
        return RuntimeShapeMaterializationRule{
            kind, runtime_shape_rule_name(kind), false};
    }
    if (find_runtime_shape_materialization_handler(kind)) {
        return RuntimeShapeMaterializationRule{
            kind, runtime_shape_rule_name(kind), true};
    }
    return std::nullopt;
}

bool runtime_shape_materialization_rule_supported(
    std::string_view runtime_shape_rule) noexcept {
    return runtime_shape_materialization_rule_for(runtime_shape_rule)
        .has_value();
}

RuntimeShapeMaterializationResult materialize_runtime_output_shapes(
    const RuntimeShapeMaterializationRequest& request) {
    const auto* descriptor = request.descriptor;
    OPENVINO_ASSERT(descriptor,
                    request.error_prefix,
                    ": compiler-owned runtime stage descriptor is required "
                    "for runtime shape materialization");
    const std::string stage_name =
        !request.stage_name.empty()
            ? std::string(request.stage_name)
            : (!descriptor->stage_name.empty()
                   ? descriptor->stage_name
                   : std::string("<unnamed>"));
    const std::string_view runtime_shape_rule(
        descriptor->runtime_shape_rule.data(),
        descriptor->runtime_shape_rule.size());
    const auto rule =
        runtime_shape_materialization_rule_for(runtime_shape_rule);
    OPENVINO_ASSERT(rule,
                    request.error_prefix,
                    ": unsupported compiler-owned runtime shape rule '",
                    std::string(runtime_shape_rule),
                    "' for stage ",
                    stage_name);
    OPENVINO_ASSERT(descriptor_owns_runtime_shape_rule(
                        descriptor->op_family, descriptor->runtime_shape_rule),
                    request.error_prefix,
                    ": runtime descriptor shape rule '",
                    descriptor->runtime_shape_rule,
                    "' does not match op family '",
                    descriptor->op_family,
                    "' for stage ",
                    stage_name);

    RuntimeShapeMaterializationResult result;
    result.kind = rule->kind;
    if (rule->kind == RuntimeShapeRuleKind::StaticOrDescriptor) {
        RuntimeInputResolver inputs = request.inputs;
        inputs.descriptor = descriptor;
        for (size_t out_idx = 0; out_idx < request.outputs.size(); ++out_idx) {
            inputs.ensure_output_shape(out_idx, request.outputs[out_idx]);
        }
        result.materialized = true;
        return result;
    }

    OPENVINO_ASSERT(rule && rule->requires_descriptor,
                    request.error_prefix,
                    ": unsupported compiler-owned runtime shape rule '",
                    std::string(runtime_shape_rule),
                    "' at stage ",
                    stage_name);
    const auto* handler =
        find_runtime_shape_materialization_handler(rule->kind);
    OPENVINO_ASSERT(handler && handler->materialize,
                    request.error_prefix,
                    ": runtime shape materializer is missing for descriptor "
                    "rule '",
                    std::string(runtime_shape_rule),
                    "' at stage ",
                    stage_name);

    RuntimeShapeMaterializationContext context;
    context.inputs = request.inputs;
    context.inputs.descriptor = descriptor;
    context.descriptor = descriptor;
    context.outputs = request.outputs;
    context.stage_name = stage_name;
    context.error_prefix = request.error_prefix;
    context.result.kind = rule->kind;
    handler->materialize(context);
    context.result.materialized = true;
    return context.result;
}

void materialize_runtime_stage_output_shapes(InferStage& stage,
                                             const std::vector<GpuTensor*>& inputs,
                                             const char* error_prefix) {
    const std::string stage_name = stage_runtime_shape_name(stage);
    RuntimeShapeMaterializationRequest request;
    request.inputs.inputs = &inputs;
    request.descriptor = runtime_stage_descriptor_or_null(stage);
    request.outputs = stage_output_refs(stage);
    request.stage_name = stage_name;
    request.error_prefix = error_prefix;
    (void)materialize_runtime_output_shapes(request);
}

}  // namespace gfx_plugin
}  // namespace ov
