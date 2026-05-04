// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <string>
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
        if (stage.stage.builder_symbol.empty()) {
            return fail("stage " + std::to_string(stage_index) + " has no builder symbol");
        }
        if (stage.stage_record_key.empty()) {
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

class GfxMpsrtStageGraphBuilder {
public:
    explicit GfxMpsrtStageGraphBuilder(std::string record_key)
        : m_record_key(std::move(record_key)) {}

    GfxMpsrtValue add_external_input(const GfxMpsrtTensorDesc& desc) {
        const auto value = next_value();
        m_inputs.push_back(desc);
        m_external_roles.push_back(GfxMpsrtExternalBufferRole::TensorInput);
        return value;
    }

    std::vector<GfxMpsrtValue> add_stage(const GfxMpsrtStageDesc& stage,
                                         std::vector<GfxMpsrtValue> inputs,
                                         std::vector<GfxMpsrtTensorDesc> output_descs) {
        std::vector<GfxMpsrtValue> outputs;
        outputs.reserve(output_descs.size());
        for (size_t i = 0; i < output_descs.size(); ++i) {
            outputs.push_back(next_value());
        }
        add_stage_with_outputs(stage, std::move(inputs), outputs, std::move(output_descs));
        return outputs;
    }

    void add_stage_with_outputs(const GfxMpsrtStageDesc& stage,
                                std::vector<GfxMpsrtValue> inputs,
                                std::vector<GfxMpsrtValue> outputs,
                                std::vector<GfxMpsrtTensorDesc> output_descs) {
        m_stages.push_back(GfxMpsrtBuilderStageSpec{stage,
                                                    gfx_mpsrt_stage_record_key(stage),
                                                    std::move(inputs),
                                                    std::move(outputs),
                                                    std::move(output_descs)});
    }

    void expose_external_output(GfxMpsrtValue value) {
        m_output_values.push_back(value);
        m_external_roles.push_back(GfxMpsrtExternalBufferRole::TensorOutput);
    }

    GfxMpsrtProgram build() const {
        GfxMpsrtProgram program{};
        program.record_key = m_record_key;
        program.multi_stage = m_stages.size() > 1;
        program.inputs = m_inputs;
        program.stages = m_stages;
        program.output_values = m_output_values;
        if (!m_external_roles.empty()) {
            program.external_buffer_abi.valid = true;
            program.external_buffer_abi.has_buffer_count = true;
            program.external_buffer_abi.has_output_buffer_count = true;
            program.external_buffer_abi.has_buffer_roles = true;
            program.external_buffer_abi.buffer_count = static_cast<uint32_t>(m_external_roles.size());
            program.external_buffer_abi.output_buffer_count =
                static_cast<uint32_t>(std::count(m_external_roles.begin(),
                                                 m_external_roles.end(),
                                                 GfxMpsrtExternalBufferRole::TensorOutput));
            program.external_buffer_abi.buffer_roles = m_external_roles;
        }
        program.valid = gfx_mpsrt_validate_program(program, nullptr);
        return program;
    }

private:
    GfxMpsrtValue next_value() {
        return m_next_value++;
    }

    std::string m_record_key;
    GfxMpsrtValue m_next_value = 0;
    std::vector<GfxMpsrtTensorDesc> m_inputs;
    std::vector<GfxMpsrtBuilderStageSpec> m_stages;
    std::vector<GfxMpsrtValue> m_output_values;
    std::vector<GfxMpsrtExternalBufferRole> m_external_roles;
};

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
                                          stage.output_descs,
                                          stage.stage_record_key);
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
