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

const GfxMpsrtTensorAbiDesc* find_tensor_desc(const std::vector<MpsrtRuntimeTensor>& tensors,
                                              GfxMpsrtValue value) {
    for (const auto& tensor : tensors) {
        if (tensor.value == value) {
            return &tensor.desc;
        }
    }
    return nullptr;
}

bool is_const_tensor_value(const std::vector<MpsrtRuntimeTensor>& tensors,
                           GfxMpsrtValue value) {
    const auto* desc = find_tensor_desc(tensors, value);
    return desc && (desc->flags & GfxMpsrtTensorFlagConst) != 0;
}

std::vector<GfxMpsrtValue> filter_const_values(const std::vector<GfxMpsrtValue>& values,
                                               const std::vector<MpsrtRuntimeTensor>& tensors) {
    std::vector<GfxMpsrtValue> filtered;
    filtered.reserve(values.size());
    for (const auto value : values) {
        if (!is_const_tensor_value(tensors, value)) {
            filtered.push_back(value);
        }
    }
    return filtered;
}

std::string record_error(size_t record_index, const std::string& message) {
    std::ostringstream stream;
    stream << "GFX MPSRT: builder record " << record_index << ": " << message;
    return stream.str();
}

bool has_value(const std::unordered_set<GfxMpsrtValue>& values, GfxMpsrtValue value) {
    return values.find(value) != values.end();
}

bool has_value(const std::vector<GfxMpsrtValue>& values, GfxMpsrtValue value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool tensor_descs_match(const GfxMpsrtTensorAbiDesc& lhs, const GfxMpsrtTensorAbiDesc& rhs) {
    if (lhs.rank != rhs.rank ||
        lhs.dtype != rhs.dtype ||
        lhs.storage != rhs.storage ||
        lhs.layout != rhs.layout ||
        lhs.flags != rhs.flags ||
        lhs.byte_offset != rhs.byte_offset ||
        lhs.byte_length != rhs.byte_length ||
        lhs.image_width != rhs.image_width ||
        lhs.image_height != rhs.image_height ||
        lhs.image_feature_channels != rhs.image_feature_channels ||
        lhs.image_batch != rhs.image_batch ||
        lhs.matrix_rows != rhs.matrix_rows ||
        lhs.matrix_columns != rhs.matrix_columns ||
        lhs.matrix_row_bytes != rhs.matrix_row_bytes ||
        lhs.matrix_count != rhs.matrix_count ||
        lhs.alias_of != rhs.alias_of) {
        return false;
    }
    for (uint32_t i = 0; i < 8; ++i) {
        if (lhs.dims[i] != rhs.dims[i] || lhs.strides[i] != rhs.strides[i]) {
            return false;
        }
    }
    return true;
}

bool resource_table_has_tensor_value(const std::vector<MpsrtRuntimeResource>& resources,
                                     GfxMpsrtValue value) {
    return std::any_of(resources.begin(),
                       resources.end(),
                       [&](const auto& resource) {
                           return resource.has_tensor_value && resource.value == value;
                       });
}

GfxMpsrtExternalBufferRole tensor_resource_role(const MpsrtModel& model,
                                                const MpsrtRuntimeTensor& tensor) {
    if (has_value(model.external_output_values, tensor.value) ||
        has_value(model.output_values, tensor.value)) {
        return GfxMpsrtExternalBufferRole::TensorOutput;
    }
    if (has_value(model.external_input_values, tensor.value) ||
        has_value(model.input_values, tensor.value)) {
        return GfxMpsrtExternalBufferRole::TensorInput;
    }
    if ((tensor.desc.flags & GfxMpsrtTensorFlagConst) != 0) {
        return GfxMpsrtExternalBufferRole::ConstBuffer;
    }
    return GfxMpsrtExternalBufferRole::Unknown;
}

MpsrtRuntimeResourceLifetime tensor_resource_lifetime(const MpsrtModel& model,
                                                      const MpsrtRuntimeTensor& tensor) {
    if (has_value(model.external_values, tensor.value) ||
        has_value(model.input_values, tensor.value) ||
        has_value(model.output_values, tensor.value) ||
        (tensor.desc.flags & GfxMpsrtTensorFlagExternalIo) != 0) {
        return MpsrtRuntimeResourceLifetime::External;
    }
    if ((tensor.desc.flags & GfxMpsrtTensorFlagConst) != 0) {
        return MpsrtRuntimeResourceLifetime::Model;
    }
    return MpsrtRuntimeResourceLifetime::Transient;
}

void append_missing_tensor_resources(MpsrtModel& model) {
    for (const auto& tensor : model.tensors) {
        if (resource_table_has_tensor_value(model.resources, tensor.value)) {
            continue;
        }
        MpsrtRuntimeResource resource{};
        resource.resource_index = static_cast<uint32_t>(model.resources.size());
        resource.role = tensor_resource_role(model, tensor);
        resource.lifetime = tensor_resource_lifetime(model, tensor);
        resource.has_tensor_value = true;
        resource.value = tensor.value;
        resource.tensor_desc = tensor.desc;
        model.resources.push_back(resource);
    }
}

bool validate_mpsrt_model_resources(const MpsrtModel& model, std::string* error) {
    for (size_t i = 0; i < model.resources.size(); ++i) {
        const auto& resource = model.resources[i];
        if (resource.resource_index != i) {
            return fail(error, "GFX MPSRT: resource table index mismatch");
        }
        if (resource.lifetime == MpsrtRuntimeResourceLifetime::Unknown) {
            return fail(error, "GFX MPSRT: resource has unknown lifetime");
        }
        if (resource.lifetime == MpsrtRuntimeResourceLifetime::External &&
            !gfx_mpsrt_is_valid_external_buffer_role(resource.role)) {
            return fail(error, "GFX MPSRT: external resource has invalid role");
        }
        if (!resource.has_tensor_value) {
            if (resource.lifetime != MpsrtRuntimeResourceLifetime::External) {
                return fail(error, "GFX MPSRT: non-tensor resource must be external");
            }
            continue;
        }
        const auto* desc = find_tensor_desc(model.tensors, resource.value);
        if (!desc) {
            return fail(error, "GFX MPSRT: tensor resource references unknown value");
        }
        if (!tensor_descs_match(resource.tensor_desc, *desc)) {
            return fail(error, "GFX MPSRT: tensor resource descriptor does not match model tensor");
        }
        if (resource.lifetime == MpsrtRuntimeResourceLifetime::Model &&
            (resource.tensor_desc.flags & GfxMpsrtTensorFlagConst) == 0) {
            return fail(error, "GFX MPSRT: model resource must be backed by a const tensor");
        }
        if (resource.lifetime == MpsrtRuntimeResourceLifetime::Transient &&
            ((resource.tensor_desc.flags & GfxMpsrtTensorFlagConst) != 0 ||
             (resource.tensor_desc.flags & GfxMpsrtTensorFlagExternalIo) != 0)) {
            return fail(error, "GFX MPSRT: transient resource has external or const tensor flags");
        }
    }
    for (const auto& binding : model.external_buffer_bindings) {
        if (binding.resource_index >= model.resources.size()) {
            return fail(error, "GFX MPSRT: external binding references missing resource");
        }
        const auto& resource = model.resources[binding.resource_index];
        if (resource.resource_index != binding.resource_index ||
            resource.arg_index != binding.arg_index ||
            resource.lifetime != MpsrtRuntimeResourceLifetime::External) {
            return fail(error, "GFX MPSRT: external binding resource contract mismatch");
        }
    }
    return true;
}

}  // namespace

bool finalize_mpsrt_model_resources(MpsrtModel& model, std::string* error) {
    append_missing_tensor_resources(model);
    return validate_mpsrt_model_resources(model, error);
}

namespace {

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
    if (record.kernel_buffer_order.empty()) {
        return fail(error, record_error(record_index, "MSL dispatch kernel buffer order is not materialized"));
    }
    if (record.kernel_buffer_order.size() !=
        static_cast<size_t>(record.msl_dispatch_desc.input_count + record.msl_dispatch_desc.output_count)) {
        return fail(error, record_error(record_index, "MSL dispatch kernel buffer order metadata mismatch"));
    }
    if (record.msl_dispatch_desc.threads_per_threadgroup == 0) {
        return fail(error, record_error(record_index, "MSL dispatch threadgroup size is not set"));
    }
    return true;
}

bool normalize_external_buffer_roles(std::vector<GfxMpsrtExternalBufferRole>& roles,
                                     uint32_t arg_count,
                                     uint32_t output_arg_count,
                                     size_t semantic_input_count,
                                     size_t semantic_output_count,
                                     std::string* error) {
    if (arg_count == 0) {
        return true;
    }
    if (output_arg_count > arg_count) {
        return fail(error, "GFX MPSRT: output arg count exceeds kernel arg count");
    }
    if (!roles.empty() && roles.size() != arg_count) {
        return fail(error, "GFX MPSRT: external buffer role count does not match kernel arg count");
    }

    if (roles.empty()) {
        const size_t semantic_external_count = semantic_input_count + semantic_output_count;
        const bool semantic_external_abi = arg_count == semantic_external_count ||
                                           (arg_count == semantic_input_count &&
                                            semantic_input_count == semantic_output_count);
        if (!semantic_external_abi) {
            return fail(error, "GFX MPSRT: external buffer ABI cannot be inferred without explicit roles");
        }

        const uint32_t input_arg_count = arg_count == semantic_external_count
                                             ? static_cast<uint32_t>(semantic_input_count)
                                             : static_cast<uint32_t>(semantic_input_count);
        roles.assign(input_arg_count, GfxMpsrtExternalBufferRole::TensorInput);
        roles.insert(roles.end(),
                     arg_count - input_arg_count,
                     GfxMpsrtExternalBufferRole::TensorOutput);
    }

    uint32_t role_output_count = 0;
    for (const auto role : roles) {
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
    return true;
}

bool materialize_multi_stage_external_buffer_abi(MpsrtModel& model,
                                                 uint32_t arg_count,
                                                 uint32_t output_arg_count,
                                                 std::string* error) {
    if (arg_count == 0) {
        return true;
    }

    if (model.semantic_input_values.empty() && !model.input_values.empty()) {
        model.semantic_input_values = model.input_values;
    }
    if (model.semantic_output_values.empty() && !model.output_values.empty()) {
        model.semantic_output_values = model.output_values;
    }

    auto roles = model.external_buffer_roles;
    if (!normalize_external_buffer_roles(roles,
                                         arg_count,
                                         output_arg_count,
                                         model.semantic_input_values.size(),
                                         model.semantic_output_values.size(),
                                         error)) {
        return false;
    }

    std::vector<GfxMpsrtValue> external_values;
    std::vector<GfxMpsrtValue> external_input_values;
    std::vector<GfxMpsrtValue> external_output_values;
    std::vector<MpsrtRuntimeResource> resources;
    std::vector<MpsrtExternalBufferBinding> external_buffer_bindings;
    external_values.reserve(roles.size());
    external_input_values.reserve(model.semantic_input_values.size());
    external_output_values.reserve(model.semantic_output_values.size());
    resources.reserve(roles.size());
    external_buffer_bindings.reserve(roles.size());

    size_t next_input = 0;
    size_t next_output = 0;
    for (size_t arg_index = 0; arg_index < roles.size(); ++arg_index) {
        const auto role = roles[arg_index];
        MpsrtRuntimeResource resource{};
        resource.resource_index = static_cast<uint32_t>(resources.size());
        resource.role = role;
        resource.lifetime = MpsrtRuntimeResourceLifetime::External;
        resource.arg_index = static_cast<uint32_t>(arg_index);
        switch (role) {
            case GfxMpsrtExternalBufferRole::TensorInput:
                if (next_input >= model.semantic_input_values.size()) {
                    return fail(error, "GFX MPSRT: external input role has no semantic input value");
                }
                resource.has_tensor_value = true;
                resource.value = model.semantic_input_values[next_input];
                if (const auto* desc = find_tensor_desc(model.tensors, resource.value)) {
                    resource.tensor_desc = *desc;
                }
                external_values.push_back(resource.value);
                external_input_values.push_back(resource.value);
                ++next_input;
                break;
            case GfxMpsrtExternalBufferRole::TensorOutput:
                if (next_output >= model.semantic_output_values.size()) {
                    return fail(error, "GFX MPSRT: external output role has no semantic output value");
                }
                resource.has_tensor_value = true;
                resource.value = model.semantic_output_values[next_output];
                if (const auto* desc = find_tensor_desc(model.tensors, resource.value)) {
                    resource.tensor_desc = *desc;
                }
                external_values.push_back(resource.value);
                external_output_values.push_back(resource.value);
                ++next_output;
                break;
            case GfxMpsrtExternalBufferRole::ConstBuffer:
            case GfxMpsrtExternalBufferRole::RuntimeParams:
            case GfxMpsrtExternalBufferRole::Metadata:
                break;
            case GfxMpsrtExternalBufferRole::Unknown:
            default:
                return fail(error, "GFX MPSRT: external buffer role is unknown");
        }
        external_buffer_bindings.push_back({resource.arg_index, resource.resource_index});
        resources.push_back(resource);
    }
    if (next_input != model.semantic_input_values.size() ||
        next_output != model.semantic_output_values.size()) {
        return fail(error, "GFX MPSRT: external buffer ABI does not cover semantic model IO");
    }

    model.external_buffer_roles = std::move(roles);
    model.resources = std::move(resources);
    model.external_buffer_bindings = std::move(external_buffer_bindings);
    model.external_values = std::move(external_values);
    model.external_input_values = std::move(external_input_values);
    model.external_output_values = std::move(external_output_values);
    return finalize_mpsrt_model_resources(model, error);
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
    stage.pool2d_desc = record.pool2d_desc;
    stage.resize2d_desc = record.resize2d_desc;
    stage.softmax_desc = record.softmax_desc;
    stage.topk_desc = record.topk_desc;
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
    model.resources.clear();
    model.external_buffer_bindings.clear();
    model.storage_bridges = plan.storage_bridges;

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
    for (const auto& bridge : model.storage_bridges) {
        if (!has_value(known_values, bridge.value)) {
            return fail(error, "GFX MPSRT: storage bridge references unknown tensor value");
        }
        GfxMpsrtStorageBridgeDesc normalized{};
        if (!gfx_mpsrt_make_storage_bridge_desc(bridge.value, bridge.tensor, bridge.direction, normalized)) {
            return fail(error, "GFX MPSRT: storage bridge contract is invalid");
        }
        if (normalized.source_storage != bridge.source_storage ||
            normalized.target_storage != bridge.target_storage) {
            return fail(error, "GFX MPSRT: storage bridge source/target storage mismatch");
        }
    }

    model.input_values = filter_const_values(model.input_values, model.tensors);
    model.semantic_input_values = filter_const_values(model.semantic_input_values, model.tensors);
    model.external_input_values = filter_const_values(model.external_input_values, model.tensors);
    model.external_values = model.external_input_values;
    model.external_values.insert(model.external_values.end(),
                                 model.external_output_values.begin(),
                                 model.external_output_values.end());

    return finalize_mpsrt_model_resources(model, error);
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
    stage.pool2d_desc = desc.pool2d_desc;
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

const MpsrtRuntimeResource* find_mpsrt_external_resource(const MpsrtModel& model,
                                                        const MpsrtExternalBufferBinding& binding) {
    if (binding.resource_index >= model.resources.size()) {
        return nullptr;
    }
    const auto& resource = model.resources[binding.resource_index];
    if (resource.resource_index != binding.resource_index || resource.arg_index != binding.arg_index ||
        resource.lifetime != MpsrtRuntimeResourceLifetime::External) {
        return nullptr;
    }
    return &resource;
}

const MpsrtRuntimeResource* find_mpsrt_resource_for_value(const MpsrtModel& model,
                                                         GfxMpsrtValue value) {
    for (const auto& resource : model.resources) {
        if (resource.has_tensor_value && resource.value == value) {
            return &resource;
        }
    }
    return nullptr;
}

bool mpsrt_model_has_external_resource_entries(const MpsrtModel& model) {
    return std::any_of(model.external_buffer_bindings.begin(),
                       model.external_buffer_bindings.end(),
                       [&](const auto& binding) {
                           const auto* resource = find_mpsrt_external_resource(model, binding);
                           return resource && !resource->has_tensor_value;
                       });
}

size_t mpsrt_model_external_buffer_abi_count(const MpsrtModel& model) {
    return model.external_buffer_bindings.empty() ? model.external_values.size()
                                                 : model.external_buffer_bindings.size();
}

size_t mpsrt_model_resource_lifetime_count(const MpsrtModel& model,
                                           MpsrtRuntimeResourceLifetime lifetime) {
    return static_cast<size_t>(std::count_if(model.resources.begin(),
                                             model.resources.end(),
                                             [&](const auto& resource) {
                                                 return resource.lifetime == lifetime;
                                             }));
}

bool adapt_mpsrt_model_to_external_buffer_abi(MpsrtModel& model,
                                              uint32_t arg_count,
                                              uint32_t output_arg_count,
                                              std::string* error) {
    if (model.stages.size() != 1 || model.stages.front().kind != GfxMpsrtStageKind::MSLDispatch) {
        return materialize_multi_stage_external_buffer_abi(model, arg_count, output_arg_count, error);
    }
    if (arg_count == 0) {
        return true;
    }

    if (model.semantic_input_values.empty() && !model.input_values.empty()) {
        model.semantic_input_values = model.input_values;
    }
    if (model.semantic_output_values.empty() && !model.output_values.empty()) {
        model.semantic_output_values = model.output_values;
    }

    if (output_arg_count > arg_count) {
        return fail(error, "GFX MPSRT: output arg count exceeds kernel arg count");
    }
    if (!model.external_buffer_roles.empty() && model.external_buffer_roles.size() != arg_count) {
        return fail(error, "GFX MPSRT: external buffer role count does not match kernel arg count");
    }
    const bool has_explicit_roles = model.external_buffer_roles.size() == arg_count;
    uint32_t role_output_count = 0;
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
        model.resources.clear();
        model.resources.reserve(model.external_buffer_roles.size());
        model.external_buffer_bindings.clear();
        model.external_buffer_bindings.reserve(model.external_buffer_roles.size());
        for (size_t i = 0; i < model.external_buffer_roles.size(); ++i) {
            MpsrtRuntimeResource resource{};
            resource.resource_index = static_cast<uint32_t>(model.resources.size());
            resource.role = model.external_buffer_roles[i];
            resource.lifetime = MpsrtRuntimeResourceLifetime::External;
            resource.arg_index = static_cast<uint32_t>(i);
            if (i < model.external_values.size()) {
                resource.has_tensor_value = true;
                resource.value = model.external_values[i];
                if (const auto* desc = find_tensor_desc(model.tensors, resource.value)) {
                    resource.tensor_desc = *desc;
                }
            }
            model.external_buffer_bindings.push_back({resource.arg_index, resource.resource_index});
            model.resources.push_back(resource);
        }
        return finalize_mpsrt_model_resources(model, error);
    }

    const uint32_t input_arg_count = has_explicit_roles ? (arg_count - role_output_count)
                                                       : (arg_count - output_arg_count);
    model.tensors.clear();
    model.input_values.clear();
    model.output_values.clear();
    model.external_values.clear();
    model.external_input_values.clear();
    model.external_output_values.clear();
    model.resources.clear();
    model.external_buffer_bindings.clear();
    model.storage_bridges.clear();
    if (!has_explicit_roles) {
        model.external_buffer_roles.clear();
    }
    model.tensors.reserve(arg_count);
    model.input_values.reserve(input_arg_count);
    model.output_values.reserve(has_explicit_roles ? role_output_count : output_arg_count);
    model.external_values.reserve(arg_count);
    model.external_buffer_roles.reserve(arg_count);
    model.resources.reserve(arg_count);

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
        MpsrtRuntimeResource resource{};
        resource.resource_index = static_cast<uint32_t>(model.resources.size());
        resource.role = role;
        resource.lifetime = MpsrtRuntimeResourceLifetime::External;
        resource.arg_index = i;
        resource.has_tensor_value = true;
        resource.value = value;
        resource.tensor_desc = desc;
        model.external_buffer_bindings.push_back({resource.arg_index, resource.resource_index});
        model.resources.push_back(resource);
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
    return finalize_mpsrt_model_resources(model, error);
}

}  // namespace mpsrt
}  // namespace metal
}  // namespace gfx_plugin
}  // namespace ov
