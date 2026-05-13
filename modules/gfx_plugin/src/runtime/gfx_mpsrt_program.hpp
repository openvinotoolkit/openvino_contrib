// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "runtime/gfx_mpsrt_builder_plan.hpp"
#include "runtime/gfx_mpsrt_plan.hpp"
#include "runtime/gfx_mpsrt_storage_bridge.hpp"

namespace ov {
namespace gfx_plugin {

struct GfxMpsrtExternalBufferAbiPlan {
    bool valid = false;
    bool has_buffer_count = false;
    bool has_output_buffer_count = false;
    bool has_buffer_roles = false;
    uint32_t buffer_count = 0;
    uint32_t output_buffer_count = 0;
    std::vector<GfxMpsrtExternalBufferRole> buffer_roles;
};

inline uint32_t gfx_mpsrt_count_program_external_output_roles(
    const std::vector<GfxMpsrtExternalBufferRole>& roles) {
    uint32_t count = 0;
    for (const auto role : roles) {
        if (role == GfxMpsrtExternalBufferRole::TensorOutput) {
            ++count;
        }
    }
    return count;
}

inline GfxMpsrtExternalBufferAbiPlan gfx_mpsrt_make_external_buffer_abi_from_roles(
    std::vector<GfxMpsrtExternalBufferRole> roles) {
    GfxMpsrtExternalBufferAbiPlan abi{};
    abi.valid = true;
    abi.has_buffer_count = true;
    abi.has_output_buffer_count = true;
    abi.has_buffer_roles = true;
    abi.buffer_count = static_cast<uint32_t>(roles.size());
    abi.output_buffer_count = gfx_mpsrt_count_program_external_output_roles(roles);
    abi.buffer_roles = std::move(roles);
    return abi;
}

inline GfxMpsrtExternalBufferAbiPlan gfx_mpsrt_make_external_io_abi(size_t input_count,
                                                                    size_t output_count) {
    std::vector<GfxMpsrtExternalBufferRole> roles;
    roles.reserve(input_count + output_count);
    roles.insert(roles.end(), input_count, GfxMpsrtExternalBufferRole::TensorInput);
    roles.insert(roles.end(), output_count, GfxMpsrtExternalBufferRole::TensorOutput);
    return gfx_mpsrt_make_external_buffer_abi_from_roles(std::move(roles));
}

struct GfxMpsrtProgram {
    bool valid = false;
    bool multi_stage = false;
    std::string record_key;
    std::vector<GfxMpsrtTensorDesc> inputs;
    std::vector<GfxMpsrtBuilderStageSpec> stages;
    std::vector<GfxMpsrtValue> output_values;
    GfxMpsrtExternalBufferAbiPlan external_buffer_abi{};
    bool has_storage_bridges = false;
    std::vector<GfxMpsrtStorageBridgeDesc> storage_bridges;
};

struct GfxMpsrtProgramValidationResult {
    bool valid = false;
    std::string error;
};

inline bool gfx_mpsrt_program_has_value(const std::vector<GfxMpsrtValue>& values,
                                        GfxMpsrtValue value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

inline GfxMpsrtProgramValidationResult gfx_mpsrt_validate_program(const GfxMpsrtProgram& program) {
    auto fail = [](const std::string& error) {
        GfxMpsrtProgramValidationResult result{};
        result.error = error;
        return result;
    };

    if (program.record_key.empty()) {
        return fail("missing program record key");
    }
    if (program.stages.empty()) {
        return fail("program has no stages");
    }
    if (!program.multi_stage && program.stages.size() != 1) {
        return fail("single-stage program must contain exactly one stage");
    }

    std::vector<GfxMpsrtValue> materialized_values;
    materialized_values.reserve(program.inputs.size());
    for (size_t i = 0; i < program.inputs.size(); ++i) {
        materialized_values.push_back(static_cast<GfxMpsrtValue>(i));
    }
    for (size_t stage_index = 0; stage_index < program.stages.size(); ++stage_index) {
        const auto& stage = program.stages[stage_index];
        if (stage.stage.kind == GfxMpsrtStageKind::Unknown) {
            return fail("stage " + std::to_string(stage_index) + " has unknown kind");
        }
        if (!gfx_mpsrt_stage_has_builder_symbol(stage.stage.kind)) {
            return fail("stage " + std::to_string(stage_index) + " has no builder symbol");
        }
        if (gfx_mpsrt_stage_record_key(stage.stage).empty()) {
            return fail("stage " + std::to_string(stage_index) + " has no record key");
        }
        if (stage.outputs.empty()) {
            return fail("stage " + std::to_string(stage_index) + " has no outputs");
        }
        if (stage.outputs.size() != stage.output_descs.size()) {
            return fail("stage " + std::to_string(stage_index) + " output/value descriptor mismatch");
        }
        for (const auto input : stage.inputs) {
            if (!gfx_mpsrt_program_has_value(materialized_values, input)) {
                return fail("stage " + std::to_string(stage_index) + " reads a value before it is materialized");
            }
        }
        for (const auto output : stage.outputs) {
            if (gfx_mpsrt_program_has_value(materialized_values, output)) {
                return fail("stage " + std::to_string(stage_index) + " overwrites an existing value");
            }
        }
        materialized_values.insert(materialized_values.end(), stage.outputs.begin(), stage.outputs.end());
    }

    for (const auto output : program.output_values) {
        if (!gfx_mpsrt_program_has_value(materialized_values, output)) {
            return fail("program returns a value that is not materialized");
        }
    }

    GfxMpsrtProgramValidationResult result{};
    result.valid = true;
    return result;
}

inline bool gfx_mpsrt_validate_program(const GfxMpsrtProgram& program,
                                       std::string* error) {
    const auto result = gfx_mpsrt_validate_program(program);
    if (!result.valid && error) {
        *error = result.error;
    }
    return result.valid;
}

inline std::vector<GfxMpsrtValue> gfx_mpsrt_make_sequential_values(size_t count,
                                                                   GfxMpsrtValue first = 0) {
    std::vector<GfxMpsrtValue> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        values.push_back(first + static_cast<GfxMpsrtValue>(i));
    }
    return values;
}

inline void gfx_mpsrt_apply_external_buffer_abi_to_builder_plan(
    const GfxMpsrtExternalBufferAbiPlan& external_buffer_abi,
    GfxMpsrtBuilderPlan& builder_plan) {
    if (!external_buffer_abi.valid) {
        return;
    }
    builder_plan.external_buffer_abi_valid = true;
    builder_plan.external_buffer_count = external_buffer_abi.buffer_count;
    builder_plan.external_output_buffer_count = external_buffer_abi.output_buffer_count;
    builder_plan.external_buffer_roles = external_buffer_abi.buffer_roles;
}

inline bool gfx_mpsrt_build_builder_plan_from_program(const GfxMpsrtProgram& program,
                                                      GfxMpsrtBuilderPlan& out) {
    out = {};
    if (!gfx_mpsrt_validate_program(program, nullptr)) {
        return false;
    }

    if (program.multi_stage) {
        out = gfx_mpsrt_make_multi_stage_builder_plan(program.record_key,
                                                      program.inputs,
                                                      program.stages,
                                                      program.output_values);
    } else if (program.stages.size() == 1) {
        const auto& stage = program.stages.front();
        out = gfx_mpsrt_make_builder_plan(stage.stage,
                                          program.inputs,
                                          stage.output_descs);
    }
    if (!out.valid) {
        return false;
    }
    if (program.has_storage_bridges) {
        out.storage_bridges = program.storage_bridges;
    }
    gfx_mpsrt_apply_external_buffer_abi_to_builder_plan(program.external_buffer_abi, out);
    return out.valid;
}

}  // namespace gfx_plugin
}  // namespace ov
