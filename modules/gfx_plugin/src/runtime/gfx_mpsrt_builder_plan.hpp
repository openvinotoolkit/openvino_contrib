// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "runtime/gfx_mpsrt_abi.hpp"
#include "runtime/gfx_mpsrt_plan.hpp"

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
    std::string stage_record_key;
    std::string kernel_name;
    std::string dispatch_kernel_family;
    std::string dispatch_entry_point;
    uint32_t dispatch_kernel_family_id = 0;
    uint32_t dispatch_flags = GfxMpsrtMslDispatchFlagNone;
    uint32_t dispatch_threads_per_threadgroup = 0;
    bool dispatch_precompiled_kernel_required = false;
    GfxMpsrtStageKind stage_kind = GfxMpsrtStageKind::Unknown;
    GfxMpsrtValue value = 0;
    std::vector<GfxMpsrtValue> inputs;
    std::vector<GfxMpsrtValue> outputs;
    std::vector<GfxMpsrtValue> kernel_buffer_order;
    std::vector<GfxMpsrtTensorAbiDesc> tensor_descs;
    GfxMpsrtConv2DAbiDesc conv2d_desc{};
    GfxMpsrtGemmAbiDesc gemm_desc{};
    GfxMpsrtMslDispatchAbiDesc msl_dispatch_desc{};
};

struct GfxMpsrtBuilderPlan {
    bool valid = false;
    bool external_buffer_abi_valid = false;
    std::string stage_record_key;
    std::vector<GfxMpsrtBuilderRecord> records;
    std::vector<GfxMpsrtValue> input_values;
    std::vector<GfxMpsrtValue> output_values;
    uint32_t external_buffer_count = 0;
    uint32_t external_output_buffer_count = 0;
    std::vector<GfxMpsrtExternalBufferRole> external_buffer_roles;
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

inline bool gfx_mpsrt_stage_uses_gemm_desc(GfxMpsrtStageKind kind) {
    return kind == GfxMpsrtStageKind::MPSGemm;
}

inline bool gfx_mpsrt_stage_uses_conv2d_desc(GfxMpsrtStageKind kind) {
    return kind == GfxMpsrtStageKind::MPSConv2D ||
           kind == GfxMpsrtStageKind::MPSGroupConv2D;
}

inline GfxMpsrtMslDispatchAbiDesc gfx_mpsrt_make_msl_dispatch_desc(const GfxMpsrtStageDesc& stage,
                                                                   uint32_t input_count,
                                                                   uint32_t output_count) {
    GfxMpsrtMslDispatchAbiDesc desc{};
    desc.kernel_family = stage.dispatch_kernel_family_id;
    desc.storage = static_cast<uint32_t>(stage.output_storage);
    desc.layout = static_cast<uint32_t>(stage.layout);
    desc.threads_per_threadgroup = std::max<uint32_t>(1u, stage.dispatch_threads_per_threadgroup);
    desc.input_count = input_count;
    desc.output_count = output_count;
    desc.flags = stage.dispatch_flags;
    return desc;
}

inline GfxMpsrtBuilderPlan gfx_mpsrt_make_builder_plan(const GfxMpsrtStageDesc& stage,
                                                       const std::vector<GfxMpsrtTensorDesc>& inputs,
                                                       const std::vector<GfxMpsrtTensorDesc>& outputs,
                                                       const std::string& stage_record_key) {
    GfxMpsrtBuilderPlan plan{};
    if (stage.kind == GfxMpsrtStageKind::Unknown ||
        stage.builder_symbol.empty() ||
        stage_record_key.empty() ||
        outputs.empty()) {
        return plan;
    }

    plan.valid = true;
    plan.stage_record_key = stage_record_key;
    plan.input_values.reserve(inputs.size());
    plan.output_values.reserve(outputs.size());

    GfxMpsrtBuilderRecord begin{};
    begin.kind = GfxMpsrtBuilderRecordKind::ModelBegin;
    begin.symbol = "ovgfx_mpsrt_model_begin";
    begin.stage_record_key = stage_record_key;
    plan.records.push_back(std::move(begin));

    GfxMpsrtValue next_value = 0;
    for (const auto& input : inputs) {
        const GfxMpsrtValue value = next_value++;
        plan.input_values.push_back(value);
        GfxMpsrtBuilderRecord add{};
        add.kind = GfxMpsrtBuilderRecordKind::AddTensor;
        add.symbol = "ovgfx_mpsrt_add_tensor";
        add.stage_record_key = stage_record_key;
        add.value = value;
        add.tensor_descs.push_back(gfx_mpsrt_to_abi_desc(input));
        plan.records.push_back(std::move(add));
    }

    GfxMpsrtBuilderRecord encode{};
    encode.kind = GfxMpsrtBuilderRecordKind::EncodeStage;
    encode.symbol = stage.builder_symbol;
    encode.stage_record_key = stage_record_key;
    encode.kernel_name = !stage.dispatch_entry_point.empty() ? stage.dispatch_entry_point : stage.kernel_name;
    encode.dispatch_kernel_family = stage.dispatch_kernel_family;
    encode.dispatch_entry_point = stage.dispatch_entry_point;
    encode.dispatch_kernel_family_id = stage.dispatch_kernel_family_id;
    encode.dispatch_flags = stage.dispatch_flags;
    encode.dispatch_threads_per_threadgroup = stage.dispatch_threads_per_threadgroup;
    encode.dispatch_precompiled_kernel_required = stage.dispatch_precompiled_kernel_required;
    encode.stage_kind = stage.kind;
    encode.inputs = plan.input_values;
    encode.outputs.reserve(outputs.size());
    encode.tensor_descs.reserve(outputs.size());
    for (const auto& output : outputs) {
        const GfxMpsrtValue value = next_value++;
        plan.output_values.push_back(value);
        encode.outputs.push_back(value);
        encode.tensor_descs.push_back(gfx_mpsrt_to_abi_desc(output));
    }
    encode.kernel_buffer_order = encode.inputs;
    encode.kernel_buffer_order.insert(encode.kernel_buffer_order.end(),
                                      encode.outputs.begin(),
                                      encode.outputs.end());
    if (stage.kind == GfxMpsrtStageKind::MSLDispatch) {
        encode.msl_dispatch_desc = gfx_mpsrt_make_msl_dispatch_desc(stage,
                                                                    static_cast<uint32_t>(inputs.size()),
                                                                    static_cast<uint32_t>(outputs.size()));
    }
    plan.records.push_back(std::move(encode));

    GfxMpsrtBuilderRecord end{};
    end.kind = GfxMpsrtBuilderRecordKind::ModelEnd;
    end.symbol = "ovgfx_mpsrt_model_end";
    end.stage_record_key = stage_record_key;
    plan.records.push_back(std::move(end));

    return plan;
}

}  // namespace gfx_plugin
}  // namespace ov
