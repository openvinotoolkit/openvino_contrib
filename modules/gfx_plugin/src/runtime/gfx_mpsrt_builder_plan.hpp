// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "runtime/gfx_mpsrt_abi.hpp"
#include "runtime/gfx_mpsrt_kernel_manifest_adapter.hpp"
#include "runtime/gfx_mpsrt_plan.hpp"
#include "runtime/gfx_mpsrt_storage_bridge.hpp"

namespace ov {
namespace gfx_plugin {

enum class GfxMpsrtBuilderRecordKind : uint32_t {
    Unknown = 0,
    ModelBegin = 1,
    AddTensor = 2,
    EncodeStage = 3,
    ModelEnd = 4,
};

struct GfxMpsrtBuilderRecord {
    GfxMpsrtBuilderRecordKind kind = GfxMpsrtBuilderRecordKind::Unknown;
    std::string symbol;
    GfxMpsrtStageDesc stage_desc{};
    GfxMpsrtValue value = 0;
    std::vector<GfxMpsrtValue> inputs;
    std::vector<GfxMpsrtValue> outputs;
    std::vector<GfxMpsrtValue> kernel_buffer_order;
    std::vector<GfxMpsrtTensorAbiDesc> tensor_descs;
};

struct GfxMpsrtBuilderPlan {
    bool valid = false;
    bool external_buffer_abi_valid = false;
    std::string model_record_key;
    std::vector<GfxMpsrtBuilderRecord> records;
    std::vector<GfxMpsrtValue> input_values;
    std::vector<GfxMpsrtValue> output_values;
    uint32_t external_buffer_count = 0;
    uint32_t external_output_buffer_count = 0;
    std::vector<GfxMpsrtExternalBufferRole> external_buffer_roles;
    std::vector<GfxMpsrtStorageBridgeDesc> storage_bridges;
};

struct GfxMpsrtBuilderStageSpec {
    GfxMpsrtStageDesc stage;
    std::vector<GfxMpsrtValue> inputs;
    std::vector<GfxMpsrtValue> outputs;
    std::vector<GfxMpsrtTensorDesc> output_descs;
};

inline const char* gfx_mpsrt_builder_record_kind_name(GfxMpsrtBuilderRecordKind kind) {
    switch (kind) {
        case GfxMpsrtBuilderRecordKind::ModelBegin:
            return "model_begin";
        case GfxMpsrtBuilderRecordKind::AddTensor:
            return "add_tensor";
        case GfxMpsrtBuilderRecordKind::EncodeStage:
            return "encode_stage";
        case GfxMpsrtBuilderRecordKind::ModelEnd:
            return "model_end";
        case GfxMpsrtBuilderRecordKind::Unknown:
        default:
            return "unknown";
    }
}

inline bool gfx_mpsrt_tensor_desc_is_const(const GfxMpsrtTensorDesc& desc) {
    return (desc.flags & GfxMpsrtTensorFlagConst) != 0;
}

inline void gfx_mpsrt_append_external_storage_bridge(std::vector<GfxMpsrtStorageBridgeDesc>& bridges,
                                                     GfxMpsrtValue value,
                                                     const GfxMpsrtTensorDesc& tensor,
                                                     bool external_output) {
    if (gfx_mpsrt_tensor_desc_is_const(tensor)) {
        return;
    }
    const auto storage = static_cast<GfxMpsrtStorage>(tensor.storage);
    const auto direction = gfx_mpsrt_external_bridge_direction_for_storage(storage, external_output);
    if (direction == GfxMpsrtStorageBridgeDirection::Unknown) {
        return;
    }
    GfxMpsrtStorageBridgeDesc bridge{};
    if (!gfx_mpsrt_make_storage_bridge_desc(value, gfx_mpsrt_to_abi_desc(tensor), direction, bridge)) {
        return;
    }
    const auto already_recorded = std::any_of(bridges.begin(), bridges.end(), [&](const auto& known) {
        return known.value == bridge.value && known.direction == bridge.direction;
    });
    if (!already_recorded) {
        bridges.push_back(bridge);
    }
}

inline GfxMpsrtMslDispatchAbiDesc gfx_mpsrt_make_msl_dispatch_desc(const GfxMpsrtStageDesc& stage,
                                                                   uint32_t input_count,
                                                                   uint32_t output_count) {
    GfxMpsrtMslDispatchAbiDesc desc{};
    const auto manifest_dispatch =
        gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(stage.stage_manifest.custom_kernel);
    if (!manifest_dispatch.valid) {
        return desc;
    }
    desc.kernel_family = manifest_dispatch.kernel_family_id;
    desc.storage = static_cast<uint32_t>(stage.output_storage);
    desc.layout = static_cast<uint32_t>(stage.layout);
    desc.threads_per_threadgroup = std::max<uint32_t>(1u, manifest_dispatch.threads_per_threadgroup);
    desc.input_count = input_count;
    desc.output_count = output_count;
    desc.flags = manifest_dispatch.flags;
    return desc;
}

inline GfxMpsrtBuilderRecord gfx_mpsrt_make_encode_stage_record(const GfxMpsrtStageDesc& stage,
                                                                const std::vector<GfxMpsrtValue>& inputs,
                                                                const std::vector<GfxMpsrtValue>& outputs,
                                                                const std::vector<GfxMpsrtTensorDesc>& output_descs) {
    GfxMpsrtBuilderRecord encode{};
    encode.kind = GfxMpsrtBuilderRecordKind::EncodeStage;
    encode.symbol = gfx_mpsrt_stage_builder_symbol(stage);
    encode.stage_desc = stage;
    const auto manifest_dispatch =
        gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(stage.stage_manifest.custom_kernel);
    if (manifest_dispatch.valid) {
        encode.stage_desc.kernel_name = manifest_dispatch.entry_point;
    }
    encode.inputs = inputs;
    encode.outputs = outputs;
    encode.tensor_descs.reserve(output_descs.size());
    for (const auto& output : output_descs) {
        encode.tensor_descs.push_back(gfx_mpsrt_to_abi_desc(output));
    }
    encode.kernel_buffer_order =
        gfx_mpsrt_kernel_buffer_order_from_kernel_abi(stage.stage_manifest.custom_kernel.external_buffer_abi,
                                                      encode.inputs,
                                                      encode.outputs);
    return encode;
}

inline GfxMpsrtBuilderPlan gfx_mpsrt_make_multi_stage_builder_plan(
    const std::string& model_record_key,
    const std::vector<GfxMpsrtTensorDesc>& inputs,
    const std::vector<GfxMpsrtBuilderStageSpec>& stages,
    std::vector<GfxMpsrtValue> output_values) {
    GfxMpsrtBuilderPlan plan{};
    if (model_record_key.empty() || stages.empty()) {
        return plan;
    }
    if (output_values.empty()) {
        output_values = stages.back().outputs;
    }
    if (output_values.empty()) {
        return plan;
    }

    plan.valid = true;
    plan.model_record_key = model_record_key;
    plan.input_values.reserve(inputs.size());
    plan.output_values = std::move(output_values);

    GfxMpsrtBuilderRecord begin{};
    begin.kind = GfxMpsrtBuilderRecordKind::ModelBegin;
    begin.symbol = "ovgfx_mpsrt_model_begin";
    plan.records.push_back(std::move(begin));

    GfxMpsrtValue next_value = 0;
    for (const auto& input : inputs) {
        const GfxMpsrtValue value = next_value++;
        plan.input_values.push_back(value);
        gfx_mpsrt_append_external_storage_bridge(plan.storage_bridges,
                                                 value,
                                                 input,
                                                 /*external_output=*/false);
        GfxMpsrtBuilderRecord add{};
        add.kind = GfxMpsrtBuilderRecordKind::AddTensor;
        add.symbol = "ovgfx_mpsrt_add_tensor";
        add.value = value;
        add.tensor_descs.push_back(gfx_mpsrt_to_abi_desc(input));
        plan.records.push_back(std::move(add));
    }

    for (const auto& spec : stages) {
        const auto stage_record_key = gfx_mpsrt_stage_record_key(spec.stage);
        if (spec.stage.kind == GfxMpsrtStageKind::Unknown ||
            !gfx_mpsrt_stage_has_builder_symbol(spec.stage.kind) ||
            stage_record_key.empty() ||
            spec.outputs.empty() ||
            spec.outputs.size() != spec.output_descs.size()) {
            return {};
        }
        plan.records.push_back(gfx_mpsrt_make_encode_stage_record(spec.stage,
                                                                  spec.inputs,
                                                                  spec.outputs,
                                                                  spec.output_descs));
        for (size_t i = 0; i < spec.outputs.size(); ++i) {
            if (std::find(plan.output_values.begin(), plan.output_values.end(), spec.outputs[i]) ==
                plan.output_values.end()) {
                continue;
            }
            gfx_mpsrt_append_external_storage_bridge(plan.storage_bridges,
                                                     spec.outputs[i],
                                                     spec.output_descs[i],
                                                     /*external_output=*/true);
        }
    }

    GfxMpsrtBuilderRecord end{};
    end.kind = GfxMpsrtBuilderRecordKind::ModelEnd;
    end.symbol = "ovgfx_mpsrt_model_end";
    plan.records.push_back(std::move(end));

    return plan;
}

inline GfxMpsrtBuilderPlan gfx_mpsrt_make_builder_plan(const GfxMpsrtStageDesc& stage,
                                                       const std::vector<GfxMpsrtTensorDesc>& inputs,
                                                       const std::vector<GfxMpsrtTensorDesc>& outputs) {
    GfxMpsrtBuilderPlan plan{};
    const auto stage_record_key = gfx_mpsrt_stage_record_key(stage);
    if (stage.kind == GfxMpsrtStageKind::Unknown ||
        !gfx_mpsrt_stage_has_builder_symbol(stage.kind) ||
        stage_record_key.empty() ||
        outputs.empty()) {
        return plan;
    }

    plan.valid = true;
    plan.model_record_key = stage_record_key;
    plan.input_values.reserve(inputs.size());
    plan.output_values.reserve(outputs.size());

    GfxMpsrtBuilderRecord begin{};
    begin.kind = GfxMpsrtBuilderRecordKind::ModelBegin;
    begin.symbol = "ovgfx_mpsrt_model_begin";
    plan.records.push_back(std::move(begin));

    GfxMpsrtValue next_value = 0;
    for (const auto& input : inputs) {
        const GfxMpsrtValue value = next_value++;
        plan.input_values.push_back(value);
        gfx_mpsrt_append_external_storage_bridge(plan.storage_bridges,
                                                 value,
                                                 input,
                                                 /*external_output=*/false);
        GfxMpsrtBuilderRecord add{};
        add.kind = GfxMpsrtBuilderRecordKind::AddTensor;
        add.symbol = "ovgfx_mpsrt_add_tensor";
        add.value = value;
        add.tensor_descs.push_back(gfx_mpsrt_to_abi_desc(input));
        plan.records.push_back(std::move(add));
    }

    std::vector<GfxMpsrtValue> stage_outputs;
    stage_outputs.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        const GfxMpsrtValue value = next_value++;
        plan.output_values.push_back(value);
        stage_outputs.push_back(value);
        gfx_mpsrt_append_external_storage_bridge(plan.storage_bridges,
                                                 value,
                                                 outputs[i],
                                                 /*external_output=*/true);
    }
    auto encode = gfx_mpsrt_make_encode_stage_record(stage,
                                                     plan.input_values,
                                                     stage_outputs,
                                                     outputs);
    plan.records.push_back(std::move(encode));

    GfxMpsrtBuilderRecord end{};
    end.kind = GfxMpsrtBuilderRecordKind::ModelEnd;
    end.symbol = "ovgfx_mpsrt_model_end";
    plan.records.push_back(std::move(end));

    return plan;
}

}  // namespace gfx_plugin
}  // namespace ov
