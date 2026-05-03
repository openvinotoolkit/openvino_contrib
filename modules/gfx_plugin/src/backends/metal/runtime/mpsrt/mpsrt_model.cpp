// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "backends/metal/runtime/mpsrt/mpsrt_model.hpp"

#include <algorithm>
#include <sstream>
#include <unordered_set>
#include <utility>

#include "openvino/core/except.hpp"
#include "runtime/gfx_mpsrt_kernel_manifest_adapter.hpp"

namespace ov {
namespace gfx_plugin {
namespace metal {
namespace mpsrt {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
    return false;
}

std::string record_error(size_t record_index, const std::string& message) {
    std::ostringstream stream;
    stream << "GFX MPSRT: builder record " << record_index << ": " << message;
    return stream.str();
}

bool has_value(const std::unordered_set<GfxMpsrtValue>& values, GfxMpsrtValue value) {
    return values.find(value) != values.end();
}

bool validate_value_list(const std::vector<GfxMpsrtValue>& values,
                         const std::unordered_set<GfxMpsrtValue>& known_values,
                         size_t record_index,
                         const char* field_name,
                         std::string* error) {
    for (const auto value : values) {
        if (!has_value(known_values, value)) {
            std::ostringstream stream;
            stream << field_name << " references unknown tensor value " << value;
            return fail(error, record_error(record_index, stream.str()));
        }
    }
    return true;
}

bool validate_msl_dispatch(const GfxMpsrtBuilderRecord& record, size_t record_index, std::string* error) {
    if (record.kernel_name.empty()) {
        return fail(error, record_error(record_index, "MSL dispatch kernel name is empty"));
    }
    if (record.dispatch_entry_point.empty()) {
        return fail(error, record_error(record_index, "MSL dispatch entry point is empty"));
    }
    if (record.msl_dispatch_desc.kernel_family == 0) {
        return fail(error, record_error(record_index, "MSL dispatch kernel family is not set"));
    }
    if (record.msl_dispatch_desc.kernel_family != record.dispatch_kernel_family_id) {
        return fail(error, record_error(record_index, "MSL dispatch kernel family metadata mismatch"));
    }
    if (record.msl_dispatch_desc.input_count != record.inputs.size()) {
        return fail(error, record_error(record_index, "MSL dispatch input count metadata mismatch"));
    }
    if (record.msl_dispatch_desc.output_count != record.outputs.size()) {
        return fail(error, record_error(record_index, "MSL dispatch output count metadata mismatch"));
    }
    if (!record.kernel_buffer_order.empty() &&
        record.kernel_buffer_order.size() !=
            static_cast<size_t>(record.msl_dispatch_desc.input_count + record.msl_dispatch_desc.output_count)) {
        return fail(error, record_error(record_index, "MSL dispatch kernel buffer order metadata mismatch"));
    }
    if (record.msl_dispatch_desc.threads_per_threadgroup == 0) {
        return fail(error, record_error(record_index, "MSL dispatch threadgroup size is not set"));
    }
    return true;
}

MpsrtRuntimeStage make_runtime_stage(const GfxMpsrtBuilderRecord& record) {
    MpsrtRuntimeStage stage{};
    stage.kind = record.stage_kind;
    stage.stage_record_key = record.stage_record_key;
    stage.kernel_name = record.kernel_name;
    stage.dispatch_kernel_family = record.dispatch_kernel_family;
    stage.dispatch_entry_point = record.dispatch_entry_point;
    stage.dispatch_kernel_family_id = record.dispatch_kernel_family_id;
    stage.dispatch_flags = record.dispatch_flags;
    stage.dispatch_threads_per_threadgroup = record.dispatch_threads_per_threadgroup;
    stage.dispatch_precompiled_kernel_required = record.dispatch_precompiled_kernel_required;
    stage.msl_dispatch_desc = record.msl_dispatch_desc;
    stage.conv2d_desc = record.conv2d_desc;
    stage.gemm_desc = record.gemm_desc;
    stage.inputs = record.inputs;
    stage.outputs = record.outputs;
    stage.kernel_buffer_order = record.kernel_buffer_order;
    stage.output_descs = record.tensor_descs;
    return stage;
}

}  // namespace

bool build_mpsrt_model_from_builder_plan(const GfxMpsrtBuilderPlan& plan,
                                         MpsrtModel& model,
                                         std::string* error) {
    model = {};

    if (!plan.valid) {
        return fail(error, "GFX MPSRT: builder plan is invalid");
    }
    if (plan.stage_record_key.empty()) {
        return fail(error, "GFX MPSRT: builder plan has empty stage record key");
    }
    if (plan.records.size() < 3) {
        return fail(error, "GFX MPSRT: builder plan does not contain begin/stage/end records");
    }
    if (plan.records.front().kind != GfxMpsrtBuilderRecordKind::ModelBegin) {
        return fail(error, "GFX MPSRT: builder plan does not start with model_begin");
    }
    if (plan.records.back().kind != GfxMpsrtBuilderRecordKind::ModelEnd) {
        return fail(error, "GFX MPSRT: builder plan does not end with model_end");
    }

    std::unordered_set<GfxMpsrtValue> known_values;
    model.stage_record_key = plan.stage_record_key;
    model.semantic_input_values = plan.input_values;
    model.semantic_output_values = plan.output_values;
    model.input_values = plan.input_values;
    model.output_values = plan.output_values;
    model.external_input_values = plan.input_values;
    model.external_output_values = plan.output_values;
    model.external_values = plan.input_values;
    model.external_values.insert(model.external_values.end(), plan.output_values.begin(), plan.output_values.end());
    model.external_buffer_roles = plan.external_buffer_roles;

    for (size_t i = 0; i < plan.records.size(); ++i) {
        const auto& record = plan.records[i];
        if (record.kind != GfxMpsrtBuilderRecordKind::EncodeStage &&
            record.stage_record_key != plan.stage_record_key) {
            return fail(error, record_error(i, "stage record key does not match builder plan"));
        }

        switch (record.kind) {
            case GfxMpsrtBuilderRecordKind::ModelBegin:
            case GfxMpsrtBuilderRecordKind::ModelEnd:
                break;
            case GfxMpsrtBuilderRecordKind::AddTensor: {
                if (record.tensor_descs.size() != 1) {
                    return fail(error, record_error(i, "add_tensor must carry exactly one tensor descriptor"));
                }
                if (!known_values.insert(record.value).second) {
                    return fail(error, record_error(i, "add_tensor redefines an existing tensor value"));
                }
                model.tensors.push_back({record.value, record.tensor_descs.front()});
                break;
            }
            case GfxMpsrtBuilderRecordKind::EncodeStage: {
                if (record.stage_record_key.empty()) {
                    return fail(error, record_error(i, "stage record key is empty"));
                }
                if (record.stage_kind == GfxMpsrtStageKind::Unknown) {
                    return fail(error, record_error(i, "stage kind is unknown"));
                }
                if (record.symbol.empty()) {
                    return fail(error, record_error(i, "stage builder symbol is empty"));
                }
                if (record.outputs.empty()) {
                    return fail(error, record_error(i, "stage has no outputs"));
                }
                if (record.tensor_descs.size() != record.outputs.size()) {
                    return fail(error, record_error(i, "stage output descriptor count mismatch"));
                }
                if (!validate_value_list(record.inputs, known_values, i, "stage inputs", error)) {
                    return false;
                }
                for (const auto output : record.outputs) {
                    if (!known_values.insert(output).second) {
                        return fail(error, record_error(i, "stage output redefines an existing tensor value"));
                    }
                }
                if (record.stage_kind == GfxMpsrtStageKind::MSLDispatch &&
                    !validate_msl_dispatch(record, i, error)) {
                    return false;
                }
                if (!validate_value_list(record.kernel_buffer_order, known_values, i, "stage kernel buffers", error)) {
                    return false;
                }
                for (size_t output_index = 0; output_index < record.outputs.size(); ++output_index) {
                    model.tensors.push_back({record.outputs[output_index], record.tensor_descs[output_index]});
                }
                model.stages.push_back(make_runtime_stage(record));
                break;
            }
            case GfxMpsrtBuilderRecordKind::Unknown:
            default:
                return fail(error, record_error(i, "unknown builder record kind"));
        }
    }

    if (model.stages.empty()) {
        return fail(error, "GFX MPSRT: builder plan produced no runtime stages");
    }
    if (!std::all_of(model.input_values.begin(), model.input_values.end(), [&](GfxMpsrtValue value) {
            return has_value(known_values, value);
        })) {
        return fail(error, "GFX MPSRT: model input list references unknown tensor values");
    }
    if (!std::all_of(model.output_values.begin(), model.output_values.end(), [&](GfxMpsrtValue value) {
            return has_value(known_values, value);
        })) {
        return fail(error, "GFX MPSRT: model output list references unknown tensor values");
    }

    return true;
}

MpsrtRuntimeStage make_mpsrt_runtime_stage_from_desc(const GfxMpsrtStageDesc& desc,
                                                     const std::string& stage_record_key,
                                                     const std::vector<GfxMpsrtValue>& inputs,
                                                     const std::vector<GfxMpsrtValue>& outputs,
                                                     const std::vector<GfxMpsrtTensorAbiDesc>& output_descs) {
    MpsrtRuntimeStage stage{};
    stage.kind = desc.kind;
    stage.stage_record_key = stage_record_key;
    stage.kernel_name = !desc.dispatch_entry_point.empty() ? desc.dispatch_entry_point : desc.kernel_name;
    stage.dispatch_kernel_family = desc.dispatch_kernel_family;
    stage.dispatch_entry_point = desc.dispatch_entry_point;
    stage.dispatch_kernel_family_id = desc.dispatch_kernel_family_id;
    stage.dispatch_flags = desc.dispatch_flags;
    stage.dispatch_threads_per_threadgroup = desc.dispatch_threads_per_threadgroup;
    stage.dispatch_precompiled_kernel_required = desc.dispatch_precompiled_kernel_required;
    stage.conv2d_desc = desc.conv2d_desc;
    stage.gemm_desc = desc.gemm_desc;
    stage.inputs = inputs;
    stage.outputs = outputs;
    stage.output_descs = output_descs;

    const auto manifest_dispatch =
        gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(desc.stage_manifest.custom_kernel);
    if (manifest_dispatch.valid) {
        stage.dispatch_kernel_family = manifest_dispatch.kernel_family;
        stage.dispatch_entry_point = manifest_dispatch.entry_point;
        stage.dispatch_kernel_family_id = manifest_dispatch.kernel_family_id;
        stage.dispatch_flags = manifest_dispatch.flags;
        stage.dispatch_threads_per_threadgroup = manifest_dispatch.threads_per_threadgroup;
        stage.dispatch_precompiled_kernel_required = manifest_dispatch.precompiled_binary_required;
        stage.kernel_name = manifest_dispatch.entry_point;
    }

    if (desc.kind == GfxMpsrtStageKind::MSLDispatch) {
        stage.msl_dispatch_desc = gfx_mpsrt_make_msl_dispatch_desc(desc,
                                                                   static_cast<uint32_t>(inputs.size()),
                                                                   static_cast<uint32_t>(outputs.size()));
        stage.kernel_buffer_order =
            gfx_mpsrt_kernel_buffer_order_from_kernel_abi(desc.stage_manifest.custom_kernel.external_buffer_abi,
                                                          stage.inputs,
                                                          stage.outputs);
    }
    return stage;
}

MpsrtModel build_mpsrt_model_from_builder_plan_or_throw(const GfxMpsrtBuilderPlan& plan) {
    MpsrtModel model;
    std::string error;
    if (!build_mpsrt_model_from_builder_plan(plan, model, &error)) {
        OPENVINO_THROW(error);
    }
    return model;
}

bool adapt_mpsrt_model_to_external_buffer_abi(MpsrtModel& model,
                                              uint32_t arg_count,
                                              uint32_t output_arg_count,
                                              std::string* error) {
    if (model.stages.size() != 1 || model.stages.front().kind != GfxMpsrtStageKind::MSLDispatch) {
        return true;
    }
    if (arg_count == 0) {
        return true;
    }
    if (output_arg_count > arg_count) {
        return fail(error, "GFX MPSRT: output arg count exceeds kernel arg count");
    }
    if (!model.external_buffer_roles.empty() && model.external_buffer_roles.size() != arg_count) {
        return fail(error, "GFX MPSRT: external buffer role count does not match kernel arg count");
    }
    uint32_t role_output_count = 0;
    const bool has_explicit_roles = model.external_buffer_roles.size() == arg_count;
    if (has_explicit_roles) {
        for (const auto role : model.external_buffer_roles) {
            if (!gfx_mpsrt_is_valid_external_buffer_role(role)) {
                return fail(error, "GFX MPSRT: external buffer role is unknown");
            }
            if (gfx_mpsrt_is_external_output_buffer_role(role)) {
                ++role_output_count;
            }
        }
        if (output_arg_count != 0 && role_output_count != output_arg_count) {
            return fail(error, "GFX MPSRT: external buffer role output count mismatch");
        }
    }

    if (model.semantic_input_values.empty() && !model.input_values.empty()) {
        model.semantic_input_values = model.input_values;
    }
    if (model.semantic_output_values.empty() && !model.output_values.empty()) {
        model.semantic_output_values = model.output_values;
    }

    const size_t semantic_external_count = model.input_values.size() + model.output_values.size();
    const bool semantic_external_abi = arg_count == semantic_external_count ||
                                       (arg_count == model.input_values.size() &&
                                        model.input_values.size() == model.output_values.size());
    if (semantic_external_abi) {
        if (model.external_values.empty()) {
            model.external_values = model.input_values;
            model.external_values.insert(model.external_values.end(),
                                         model.output_values.begin(),
                                         model.output_values.end());
        }
        if (model.external_input_values.empty()) {
            model.external_input_values = model.input_values;
        }
        if (model.external_output_values.empty()) {
            model.external_output_values = model.output_values;
        }
        if (model.external_buffer_roles.empty()) {
            model.external_buffer_roles.assign(model.input_values.size(), GfxMpsrtExternalBufferRole::TensorInput);
            model.external_buffer_roles.insert(model.external_buffer_roles.end(),
                                               model.output_values.size(),
                                               GfxMpsrtExternalBufferRole::TensorOutput);
        }
        return true;
    }

    const uint32_t input_arg_count = has_explicit_roles ? (arg_count - role_output_count)
                                                       : (arg_count - output_arg_count);
    model.tensors.clear();
    model.input_values.clear();
    model.output_values.clear();
    model.external_values.clear();
    model.external_input_values.clear();
    model.external_output_values.clear();
    if (!has_explicit_roles) {
        model.external_buffer_roles.clear();
    }
    model.tensors.reserve(arg_count);
    model.input_values.reserve(input_arg_count);
    model.output_values.reserve(has_explicit_roles ? role_output_count : output_arg_count);
    model.external_values.reserve(arg_count);
    model.external_buffer_roles.reserve(arg_count);

    for (uint32_t i = 0; i < arg_count; ++i) {
        const auto value = static_cast<GfxMpsrtValue>(i);
        const auto role = has_explicit_roles
                              ? model.external_buffer_roles[i]
                              : (i < input_arg_count ? GfxMpsrtExternalBufferRole::TensorInput
                                                     : GfxMpsrtExternalBufferRole::TensorOutput);
        GfxMpsrtTensorAbiDesc desc{};
        desc.flags = GfxMpsrtTensorFlagExternalIo;
        model.external_values.push_back(value);
        if (!has_explicit_roles) {
            model.external_buffer_roles.push_back(role);
        }
        if (gfx_mpsrt_is_external_output_buffer_role(role)) {
            model.output_values.push_back(value);
            model.external_output_values.push_back(value);
        } else {
            model.input_values.push_back(value);
            model.external_input_values.push_back(value);
        }
        model.tensors.push_back({value, desc});
    }

    auto& stage = model.stages.front();
    stage.inputs = model.input_values;
    stage.outputs = model.output_values;
    stage.output_descs.assign(stage.outputs.size(), GfxMpsrtTensorAbiDesc{});
    auto kernel_buffer_order =
        gfx_mpsrt_kernel_buffer_order_from_external_values(model.external_buffer_roles, model.external_values);
    if (kernel_buffer_order.empty()) {
        return fail(error, "GFX MPSRT: external kernel buffer order cannot be materialized");
    }
    stage.kernel_buffer_order = std::move(kernel_buffer_order);
    stage.msl_dispatch_desc.input_count = input_arg_count;
    stage.msl_dispatch_desc.output_count = output_arg_count;
    return true;
}

}  // namespace mpsrt
}  // namespace metal
}  // namespace gfx_plugin
}  // namespace ov
