// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "runtime/gfx_mpsrt_model.hpp"

#include <algorithm>
#include <sstream>
#include <unordered_set>
#include <utility>

#include "openvino/core/except.hpp"
#include "runtime/gfx_mpsrt_kernel_manifest_adapter.hpp"

namespace ov {
namespace gfx_plugin {
namespace mpsrt {
namespace {

bool fail(std::string *error, const std::string &message) {
  if (error) {
    *error = message;
  }
  return false;
}

const GfxMpsrtTensorAbiDesc *
find_tensor_desc(const std::vector<MpsrtRuntimeTensor> &tensors,
                 GfxMpsrtValue value) {
  for (const auto &tensor : tensors) {
    if (tensor.value == value) {
      return &tensor.desc;
    }
  }
  return nullptr;
}

bool is_const_tensor_value(const std::vector<MpsrtRuntimeTensor> &tensors,
                           GfxMpsrtValue value) {
  const auto *desc = find_tensor_desc(tensors, value);
  return desc && (desc->flags & GfxMpsrtTensorFlagConst) != 0;
}

std::vector<GfxMpsrtValue>
filter_const_values(const std::vector<GfxMpsrtValue> &values,
                    const std::vector<MpsrtRuntimeTensor> &tensors) {
  std::vector<GfxMpsrtValue> filtered;
  filtered.reserve(values.size());
  for (const auto value : values) {
    if (!is_const_tensor_value(tensors, value)) {
      filtered.push_back(value);
    }
  }
  return filtered;
}

std::string record_error(size_t record_index, const std::string &message) {
  std::ostringstream stream;
  stream << "GFX MPSRT: builder record " << record_index << ": " << message;
  return stream.str();
}

bool has_value(const std::unordered_set<GfxMpsrtValue> &values,
               GfxMpsrtValue value) {
  return values.find(value) != values.end();
}

bool has_value(const std::vector<GfxMpsrtValue> &values, GfxMpsrtValue value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

bool tensor_descs_match(const GfxMpsrtTensorAbiDesc &lhs,
                        const GfxMpsrtTensorAbiDesc &rhs) {
  if (lhs.rank != rhs.rank || lhs.dtype != rhs.dtype ||
      lhs.storage != rhs.storage || lhs.layout != rhs.layout ||
      lhs.flags != rhs.flags || lhs.byte_offset != rhs.byte_offset ||
      lhs.byte_length != rhs.byte_length ||
      lhs.image_width != rhs.image_width ||
      lhs.image_height != rhs.image_height ||
      lhs.image_feature_channels != rhs.image_feature_channels ||
      lhs.image_batch != rhs.image_batch ||
      lhs.matrix_rows != rhs.matrix_rows ||
      lhs.matrix_columns != rhs.matrix_columns ||
      lhs.matrix_row_bytes != rhs.matrix_row_bytes ||
      lhs.matrix_count != rhs.matrix_count || lhs.alias_of != rhs.alias_of) {
    return false;
  }
  for (uint32_t i = 0; i < 8; ++i) {
    if (lhs.dims[i] != rhs.dims[i] || lhs.strides[i] != rhs.strides[i]) {
      return false;
    }
  }
  return true;
}

bool resource_table_has_tensor_value(
    const std::vector<MpsrtRuntimeResource> &resources, GfxMpsrtValue value) {
  return std::any_of(
      resources.begin(), resources.end(), [&](const auto &resource) {
        return resource.has_tensor_value && resource.value == value;
      });
}

std::vector<GfxMpsrtValue>
collect_const_tensor_values(const MpsrtModel &model) {
  std::vector<GfxMpsrtValue> values;
  values.reserve(model.tensors.size());
  for (const auto &tensor : model.tensors) {
    if ((tensor.desc.flags & GfxMpsrtTensorFlagConst) != 0) {
      values.push_back(tensor.value);
    }
  }
  return values;
}

const std::vector<GfxMpsrtValue> *
single_stage_kernel_buffer_order_for_roles(
    const MpsrtModel &model,
    const std::vector<GfxMpsrtExternalBufferRole> &roles) {
  if (model.stages.size() != 1) {
    return nullptr;
  }
  const auto &order = model.stages.front().kernel_buffer_order;
  if (order.size() != roles.size()) {
    return nullptr;
  }
  return &order;
}

bool is_single_stage_msl_dispatch_model(const MpsrtModel &model) {
  return model.stages.size() == 1 &&
         model.stages.front().kind == GfxMpsrtStageKind::MSLDispatch;
}

void normalize_semantic_io_from_kernel_buffer_order(
    MpsrtModel &model, const std::vector<GfxMpsrtExternalBufferRole> &roles) {
  const auto *order = single_stage_kernel_buffer_order_for_roles(model, roles);
  if (!order) {
    return;
  }

  std::vector<GfxMpsrtValue> semantic_inputs;
  std::vector<GfxMpsrtValue> semantic_outputs;
  semantic_inputs.reserve(model.semantic_input_values.size());
  semantic_outputs.reserve(model.semantic_output_values.size());

  for (size_t arg_index = 0; arg_index < roles.size(); ++arg_index) {
    const auto value = (*order)[arg_index];
    switch (roles[arg_index]) {
    case GfxMpsrtExternalBufferRole::TensorInput:
      if (!is_const_tensor_value(model.tensors, value)) {
        semantic_inputs.push_back(value);
      }
      break;
    case GfxMpsrtExternalBufferRole::TensorOutput:
      semantic_outputs.push_back(value);
      break;
    case GfxMpsrtExternalBufferRole::ConstBuffer:
    case GfxMpsrtExternalBufferRole::RuntimeParams:
    case GfxMpsrtExternalBufferRole::Metadata:
    case GfxMpsrtExternalBufferRole::Unknown:
    default:
      break;
    }
  }

  if (!semantic_inputs.empty()) {
    model.semantic_input_values = std::move(semantic_inputs);
  }
  if (!semantic_outputs.empty()) {
    model.semantic_output_values = std::move(semantic_outputs);
  }
}

GfxMpsrtValue next_external_resource_value(const MpsrtModel &model) {
  GfxMpsrtValue next_value = 0;
  auto visit = [&](GfxMpsrtValue value) {
    next_value = std::max<GfxMpsrtValue>(next_value, value + 1);
  };
  for (const auto &tensor : model.tensors) {
    visit(tensor.value);
  }
  for (const auto value : model.semantic_input_values) {
    visit(value);
  }
  for (const auto value : model.semantic_output_values) {
    visit(value);
  }
  for (const auto value : model.input_values) {
    visit(value);
  }
  for (const auto value : model.output_values) {
    visit(value);
  }
  for (const auto value : model.external_values) {
    visit(value);
  }
  for (const auto &stage : model.stages) {
    for (const auto value : stage.inputs) {
      visit(value);
    }
    for (const auto value : stage.outputs) {
      visit(value);
    }
    for (const auto value : stage.kernel_buffer_order) {
      visit(value);
    }
  }
  return next_value;
}

GfxMpsrtExternalBufferRole
tensor_resource_role(const MpsrtModel &model,
                     const MpsrtRuntimeTensor &tensor) {
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

MpsrtRuntimeResourceLifetime
tensor_resource_lifetime(const MpsrtModel &model,
                         const MpsrtRuntimeTensor &tensor) {
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

const MpsrtRuntimeResource *
find_resource_for_value_and_lifetime(const MpsrtModel &model,
                                     GfxMpsrtValue value,
                                     MpsrtRuntimeResourceLifetime lifetime) {
  for (const auto &resource : model.resources) {
    if (resource.has_tensor_value && resource.value == value &&
        resource.lifetime == lifetime) {
      return &resource;
    }
  }
  return nullptr;
}

GfxMpsrtStorageBridgeDirection
external_binding_bridge_direction(const MpsrtModel &model, GfxMpsrtValue value,
                                  bool external_output) {
  const auto *desc = find_tensor_desc(model.tensors, value);
  const auto storage = desc ? static_cast<GfxMpsrtStorage>(desc->storage)
                            : GfxMpsrtStorage::Unknown;
  const auto storage_direction =
      gfx_mpsrt_external_bridge_direction_for_storage(storage, external_output);
  const auto fallback_direction =
      storage_direction == GfxMpsrtStorageBridgeDirection::Unknown
          ? gfx_mpsrt_external_image_bridge_direction(external_output)
          : storage_direction;
  if (const auto *bridge = find_mpsrt_storage_bridge(model, value)) {
    return bridge->direction;
  }
  return fallback_direction;
}

void append_missing_tensor_resources(MpsrtModel &model) {
  for (const auto &tensor : model.tensors) {
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

bool validate_mpsrt_model_resources(const MpsrtModel &model,
                                    std::string *error) {
  for (size_t i = 0; i < model.resources.size(); ++i) {
    const auto &resource = model.resources[i];
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
    const auto *desc = find_tensor_desc(model.tensors, resource.value);
    if (!desc) {
      return fail(error, "GFX MPSRT: tensor resource references unknown value");
    }
    if (!tensor_descs_match(resource.tensor_desc, *desc)) {
      return fail(
          error,
          "GFX MPSRT: tensor resource descriptor does not match model tensor");
    }
    if (resource.lifetime == MpsrtRuntimeResourceLifetime::Model &&
        (resource.tensor_desc.flags & GfxMpsrtTensorFlagConst) == 0) {
      return fail(error,
                  "GFX MPSRT: model resource must be backed by a const tensor");
    }
    if (resource.lifetime == MpsrtRuntimeResourceLifetime::Transient &&
        ((resource.tensor_desc.flags & GfxMpsrtTensorFlagConst) != 0 ||
         (resource.tensor_desc.flags & GfxMpsrtTensorFlagExternalIo) != 0)) {
      return fail(
          error,
          "GFX MPSRT: transient resource has external or const tensor flags");
    }
  }
  for (const auto &binding : model.external_buffer_bindings) {
    if (binding.resource_index >= model.resources.size()) {
      return fail(error,
                  "GFX MPSRT: external binding references missing resource");
    }
    const auto &resource = model.resources[binding.resource_index];
    if (resource.resource_index != binding.resource_index ||
        resource.arg_index != binding.arg_index ||
        resource.lifetime != MpsrtRuntimeResourceLifetime::External) {
      return fail(error,
                  "GFX MPSRT: external binding resource contract mismatch");
    }
  }
  return true;
}

} // namespace

bool finalize_mpsrt_model_resources(MpsrtModel &model, std::string *error) {
  append_missing_tensor_resources(model);
  return validate_mpsrt_model_resources(model, error);
}

namespace {

bool validate_value_list(const std::vector<GfxMpsrtValue> &values,
                         const std::unordered_set<GfxMpsrtValue> &known_values,
                         size_t record_index, const char *field_name,
                         std::string *error) {
  for (const auto value : values) {
    if (!has_value(known_values, value)) {
      std::ostringstream stream;
      stream << field_name << " references unknown tensor value " << value;
      return fail(error, record_error(record_index, stream.str()));
    }
  }
  return true;
}

bool validate_msl_dispatch(const GfxMpsrtBuilderRecord &record,
                           size_t record_index, std::string *error) {
  if (record.stage_desc.kernel_name.empty()) {
    return fail(
        error, record_error(record_index, "MSL dispatch kernel name is empty"));
  }
  const auto manifest_dispatch =
      gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(
          record.stage_desc.stage_manifest.custom_kernel);
  const auto &custom_kernel = record.stage_desc.stage_manifest.custom_kernel;
  if (custom_kernel.entry_point.empty()) {
    return fail(
        error, record_error(record_index, "MSL dispatch entry point is empty"));
  }
  if (custom_kernel.kernel_family_id == 0) {
    return fail(error, record_error(record_index,
                                    "MSL dispatch kernel family is not set"));
  }
  if (!manifest_dispatch.valid) {
    return fail(error,
                record_error(record_index,
                             "MSL dispatch manifest metadata is not set"));
  }
  const auto msl_dispatch_desc = gfx_mpsrt_make_msl_dispatch_desc(
      record.stage_desc, static_cast<uint32_t>(record.inputs.size()),
      static_cast<uint32_t>(record.outputs.size()));
  if (msl_dispatch_desc.kernel_family == 0) {
    return fail(error, record_error(record_index,
                                    "MSL dispatch kernel family is not set"));
  }
  if (msl_dispatch_desc.kernel_family != manifest_dispatch.kernel_family_id) {
    return fail(error,
                record_error(record_index,
                             "MSL dispatch kernel family metadata mismatch"));
  }
  if (msl_dispatch_desc.input_count != record.inputs.size()) {
    return fail(error,
                record_error(record_index,
                             "MSL dispatch input count metadata mismatch"));
  }
  if (msl_dispatch_desc.output_count != record.outputs.size()) {
    return fail(error,
                record_error(record_index,
                             "MSL dispatch output count metadata mismatch"));
  }
  if (record.kernel_buffer_order.empty()) {
    return fail(
        error,
        record_error(record_index,
                     "MSL dispatch kernel buffer order is not materialized"));
  }
  if (record.kernel_buffer_order.size() !=
      static_cast<size_t>(msl_dispatch_desc.input_count +
                          msl_dispatch_desc.output_count)) {
    return fail(
        error,
        record_error(record_index,
                     "MSL dispatch kernel buffer order metadata mismatch"));
  }
  if (msl_dispatch_desc.threads_per_threadgroup == 0) {
    return fail(
        error,
        record_error(record_index, "MSL dispatch threadgroup size is not set"));
  }
  return true;
}

bool normalize_external_buffer_roles(
    std::vector<GfxMpsrtExternalBufferRole> &roles, uint32_t arg_count,
    uint32_t output_arg_count, std::string *error) {
  if (arg_count == 0) {
    return true;
  }
  if (roles.empty()) {
    return fail(error,
                "GFX MPSRT: external buffer ABI requires explicit roles");
  }
  if (output_arg_count > arg_count) {
    return fail(error, "GFX MPSRT: output arg count exceeds kernel arg count");
  }
  if (roles.size() != arg_count) {
    return fail(error, "GFX MPSRT: external buffer role count does not match "
                       "kernel arg count");
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

bool materialize_multi_stage_external_buffer_abi(MpsrtModel &model,
                                                 uint32_t arg_count,
                                                 uint32_t output_arg_count,
                                                 std::string *error) {
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
  if (!normalize_external_buffer_roles(roles, arg_count, output_arg_count,
                                       error)) {
    return false;
  }
  normalize_semantic_io_from_kernel_buffer_order(model, roles);
  model.semantic_input_values =
      filter_const_values(model.semantic_input_values, model.tensors);
  model.input_values = filter_const_values(model.input_values, model.tensors);
  model.external_input_values =
      filter_const_values(model.external_input_values, model.tensors);

  std::vector<GfxMpsrtValue> external_values;
  std::vector<GfxMpsrtValue> external_input_values;
  std::vector<GfxMpsrtValue> external_output_values;
  std::vector<GfxMpsrtValue> kernel_buffer_order;
  std::vector<MpsrtRuntimeResource> resources;
  std::vector<MpsrtExternalBufferBinding> external_buffer_bindings;
  external_values.reserve(roles.size());
  external_input_values.reserve(model.semantic_input_values.size());
  external_output_values.reserve(model.semantic_output_values.size());
  kernel_buffer_order.reserve(roles.size());
  resources.reserve(roles.size());
  external_buffer_bindings.reserve(roles.size());

  size_t next_input = 0;
  size_t next_output = 0;
  size_t next_const = 0;
  const bool externalize_msl_dispatch_const_buffers =
      is_single_stage_msl_dispatch_model(model);
  const auto const_values = collect_const_tensor_values(model);
  const auto *explicit_kernel_order =
      single_stage_kernel_buffer_order_for_roles(model, roles);
  auto next_non_tensor_value = next_external_resource_value(model);
  for (size_t arg_index = 0; arg_index < roles.size(); ++arg_index) {
    const auto role = roles[arg_index];
    MpsrtRuntimeResource resource{};
    resource.role = role;
    resource.lifetime = MpsrtRuntimeResourceLifetime::External;
    resource.arg_index = static_cast<uint32_t>(arg_index);
    bool append_external_resource = true;
    switch (role) {
    case GfxMpsrtExternalBufferRole::TensorInput:
      if (next_input >= model.semantic_input_values.size()) {
        return fail(
            error,
            "GFX MPSRT: external input role has no semantic input value");
      }
      resource.has_tensor_value = true;
      resource.value = model.semantic_input_values[next_input];
      if (const auto *desc = find_tensor_desc(model.tensors, resource.value)) {
        resource.tensor_desc = *desc;
      } else {
        resource.has_tensor_value = false;
        resource.value = next_non_tensor_value++;
      }
      external_values.push_back(resource.value);
      external_input_values.push_back(resource.value);
      kernel_buffer_order.push_back(resource.value);
      ++next_input;
      break;
    case GfxMpsrtExternalBufferRole::TensorOutput:
      if (next_output >= model.semantic_output_values.size()) {
        return fail(
            error,
            "GFX MPSRT: external output role has no semantic output value");
      }
      resource.has_tensor_value = true;
      resource.value = model.semantic_output_values[next_output];
      if (const auto *desc = find_tensor_desc(model.tensors, resource.value)) {
        resource.tensor_desc = *desc;
      }
      external_values.push_back(resource.value);
      external_output_values.push_back(resource.value);
      kernel_buffer_order.push_back(resource.value);
      ++next_output;
      break;
    case GfxMpsrtExternalBufferRole::ConstBuffer:
      if (externalize_msl_dispatch_const_buffers) {
        if (explicit_kernel_order && arg_index < explicit_kernel_order->size()) {
          resource.value = (*explicit_kernel_order)[arg_index];
        } else if (next_const < const_values.size()) {
          resource.value = const_values[next_const++];
        } else {
          resource.value = next_non_tensor_value++;
        }
        if (const auto *desc = find_tensor_desc(model.tensors, resource.value)) {
          resource.has_tensor_value = true;
          resource.tensor_desc = *desc;
        }
        external_values.push_back(resource.value);
        external_input_values.push_back(resource.value);
        kernel_buffer_order.push_back(resource.value);
        break;
      }
      if (explicit_kernel_order &&
          is_const_tensor_value(model.tensors, (*explicit_kernel_order)[arg_index])) {
        kernel_buffer_order.push_back((*explicit_kernel_order)[arg_index]);
        append_external_resource = false;
        break;
      }
      if (next_const < const_values.size()) {
        kernel_buffer_order.push_back(const_values[next_const++]);
        append_external_resource = false;
        break;
      }
      if (next_input < model.semantic_input_values.size()) {
        resource.value = model.semantic_input_values[next_input];
        if (const auto *desc =
                find_tensor_desc(model.tensors, resource.value)) {
          resource.has_tensor_value = true;
          resource.tensor_desc = *desc;
        } else {
          resource.has_tensor_value = false;
          resource.value = next_non_tensor_value++;
        }
        external_values.push_back(resource.value);
        external_input_values.push_back(resource.value);
        kernel_buffer_order.push_back(resource.value);
        ++next_input;
        break;
      }
      resource.has_tensor_value = false;
      resource.value = next_non_tensor_value++;
      external_values.push_back(resource.value);
      external_input_values.push_back(resource.value);
      kernel_buffer_order.push_back(resource.value);
      break;
    case GfxMpsrtExternalBufferRole::RuntimeParams:
    case GfxMpsrtExternalBufferRole::Metadata:
      resource.has_tensor_value = false;
      resource.value = next_non_tensor_value++;
      external_values.push_back(resource.value);
      external_input_values.push_back(resource.value);
      kernel_buffer_order.push_back(resource.value);
      break;
    case GfxMpsrtExternalBufferRole::Unknown:
    default:
      return fail(error, "GFX MPSRT: external buffer role is unknown");
    }
    if (append_external_resource) {
      resource.resource_index = static_cast<uint32_t>(resources.size());
      external_buffer_bindings.push_back(
          {resource.arg_index, resource.resource_index});
      resources.push_back(resource);
    }
  }
  if (next_input != model.semantic_input_values.size() ||
      next_output != model.semantic_output_values.size()) {
    std::ostringstream stream;
    stream << "GFX MPSRT: external buffer ABI does not cover semantic model IO"
           << " (covered_inputs=" << next_input
           << ", semantic_inputs=" << model.semantic_input_values.size()
           << ", covered_outputs=" << next_output
           << ", semantic_outputs=" << model.semantic_output_values.size()
           << ", roles=" << roles.size();
    if (model.stages.size() == 1) {
      stream << ", kernel_order="
             << model.stages.front().kernel_buffer_order.size()
             << ", stage_inputs=" << model.stages.front().inputs.size()
             << ", stage_outputs=" << model.stages.front().outputs.size();
    }
    stream << ")";
    return fail(error, stream.str());
  }

  model.external_buffer_roles = std::move(roles);
  model.resources = std::move(resources);
  model.external_buffer_bindings = std::move(external_buffer_bindings);
  model.external_values = std::move(external_values);
  model.external_input_values = std::move(external_input_values);
  model.external_output_values = std::move(external_output_values);
  model.input_values = model.external_input_values;
  model.output_values = model.external_output_values;

  if (model.stages.size() == 1 &&
      model.stages.front().kind == GfxMpsrtStageKind::MSLDispatch) {
    auto &stage = model.stages.front();
    stage.inputs = model.external_input_values;
    stage.outputs = model.external_output_values;
    stage.kernel_buffer_order = kernel_buffer_order;
    if (stage.kernel_buffer_order.empty() ||
        stage.kernel_buffer_order.size() !=
            model.external_buffer_roles.size()) {
      return fail(
          error,
          "GFX MPSRT: external buffer ABI cannot materialize MSL buffer order");
    }
    stage.msl_dispatch_desc.input_count = static_cast<uint32_t>(
        stage.kernel_buffer_order.size() - stage.outputs.size());
    stage.msl_dispatch_desc.output_count =
        static_cast<uint32_t>(stage.outputs.size());
  }
  return finalize_mpsrt_model_resources(model, error);
}

MpsrtRuntimeStage make_runtime_stage(const GfxMpsrtBuilderRecord &record) {
  return make_mpsrt_runtime_stage_from_desc(
      record.stage_desc, record.inputs, record.outputs, record.tensor_descs);
}

} // namespace

bool build_mpsrt_model_from_builder_plan(const GfxMpsrtBuilderPlan &plan,
                                         MpsrtModel &model,
                                         std::string *error) {
  model = {};

  if (!plan.valid) {
    return fail(error, "GFX MPSRT: builder plan is invalid");
  }
  if (plan.model_record_key.empty()) {
    return fail(error, "GFX MPSRT: builder plan has empty model record key");
  }
  if (plan.records.size() < 3) {
    return fail(
        error,
        "GFX MPSRT: builder plan does not contain begin/stage/end records");
  }
  if (plan.records.front().kind != GfxMpsrtBuilderRecordKind::ModelBegin) {
    return fail(error,
                "GFX MPSRT: builder plan does not start with model_begin");
  }
  if (plan.records.back().kind != GfxMpsrtBuilderRecordKind::ModelEnd) {
    return fail(error, "GFX MPSRT: builder plan does not end with model_end");
  }

  std::unordered_set<GfxMpsrtValue> known_values;
  model.stage_record_key = plan.model_record_key;
  model.semantic_input_values = plan.input_values;
  model.semantic_output_values = plan.output_values;
  model.input_values = plan.input_values;
  model.output_values = plan.output_values;
  model.external_input_values = plan.input_values;
  model.external_output_values = plan.output_values;
  model.external_values = plan.input_values;
  model.external_values.insert(model.external_values.end(),
                               plan.output_values.begin(),
                               plan.output_values.end());
  model.external_buffer_roles = plan.external_buffer_roles;
  model.resources.clear();
  model.external_buffer_bindings.clear();
  model.storage_bridges = plan.storage_bridges;

  for (size_t i = 0; i < plan.records.size(); ++i) {
    const auto &record = plan.records[i];
    switch (record.kind) {
    case GfxMpsrtBuilderRecordKind::ModelBegin:
    case GfxMpsrtBuilderRecordKind::ModelEnd:
      break;
    case GfxMpsrtBuilderRecordKind::AddTensor: {
      if (record.tensor_descs.size() != 1) {
        return fail(
            error,
            record_error(
                i, "add_tensor must carry exactly one tensor descriptor"));
      }
      if (!known_values.insert(record.value).second) {
        return fail(
            error,
            record_error(i, "add_tensor redefines an existing tensor value"));
      }
      model.tensors.push_back({record.value, record.tensor_descs.front()});
      break;
    }
    case GfxMpsrtBuilderRecordKind::EncodeStage: {
      const auto stage_record_key =
          gfx_mpsrt_stage_record_key(record.stage_desc);
      if (stage_record_key.empty()) {
        return fail(error, record_error(i, "stage record key is empty"));
      }
      if (record.stage_desc.kind == GfxMpsrtStageKind::Unknown) {
        return fail(error, record_error(i, "stage kind is unknown"));
      }
      if (record.symbol.empty()) {
        return fail(error, record_error(i, "stage builder symbol is empty"));
      }
      if (record.symbol != gfx_mpsrt_stage_builder_symbol(record.stage_desc)) {
        return fail(error, record_error(i, "stage builder symbol mismatch"));
      }
      if (record.outputs.empty()) {
        return fail(error, record_error(i, "stage has no outputs"));
      }
      if (record.tensor_descs.size() != record.outputs.size()) {
        return fail(error,
                    record_error(i, "stage output descriptor count mismatch"));
      }
      if (!validate_value_list(record.inputs, known_values, i, "stage inputs",
                               error)) {
        return false;
      }
      for (const auto output : record.outputs) {
        if (!known_values.insert(output).second) {
          return fail(
              error, record_error(
                         i, "stage output redefines an existing tensor value"));
        }
      }
      if (record.stage_desc.kind == GfxMpsrtStageKind::MSLDispatch &&
          !validate_msl_dispatch(record, i, error)) {
        return false;
      }
      if (!validate_value_list(record.kernel_buffer_order, known_values, i,
                               "stage kernel buffers", error)) {
        return false;
      }
      for (size_t output_index = 0; output_index < record.outputs.size();
           ++output_index) {
        model.tensors.push_back(
            {record.outputs[output_index], record.tensor_descs[output_index]});
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
  if (!std::all_of(model.input_values.begin(), model.input_values.end(),
                   [&](GfxMpsrtValue value) {
                     return has_value(known_values, value);
                   })) {
    return fail(error,
                "GFX MPSRT: model input list references unknown tensor values");
  }
  if (!std::all_of(model.output_values.begin(), model.output_values.end(),
                   [&](GfxMpsrtValue value) {
                     return has_value(known_values, value);
                   })) {
    return fail(
        error, "GFX MPSRT: model output list references unknown tensor values");
  }
  for (const auto &bridge : model.storage_bridges) {
    if (!has_value(known_values, bridge.value)) {
      return fail(error,
                  "GFX MPSRT: storage bridge references unknown tensor value");
    }
    GfxMpsrtStorageBridgeDesc normalized{};
    if (!gfx_mpsrt_make_storage_bridge_desc(bridge.value, bridge.tensor,
                                            bridge.direction, normalized)) {
      return fail(error, "GFX MPSRT: storage bridge contract is invalid");
    }
    if (normalized.source_storage != bridge.source_storage ||
        normalized.target_storage != bridge.target_storage) {
      return fail(error,
                  "GFX MPSRT: storage bridge source/target storage mismatch");
    }
  }

  model.input_values = filter_const_values(model.input_values, model.tensors);
  model.semantic_input_values =
      filter_const_values(model.semantic_input_values, model.tensors);
  model.external_input_values =
      filter_const_values(model.external_input_values, model.tensors);
  model.external_values = model.external_input_values;
  model.external_values.insert(model.external_values.end(),
                               model.external_output_values.begin(),
                               model.external_output_values.end());

  return finalize_mpsrt_model_resources(model, error);
}

MpsrtRuntimeStage make_mpsrt_runtime_stage_from_desc(
    const GfxMpsrtStageDesc &desc, const std::vector<GfxMpsrtValue> &inputs,
    const std::vector<GfxMpsrtValue> &outputs,
    const std::vector<GfxMpsrtTensorAbiDesc> &output_descs) {
  MpsrtRuntimeStage stage{};
  stage.kind = desc.kind;
  stage.stage_record_key = gfx_mpsrt_stage_record_key(desc);
  stage.kernel_name = desc.kernel_name;
  stage.conv2d_desc = desc.conv2d_desc;
  stage.gemm_desc = desc.gemm_desc;
  stage.pool2d_desc = desc.pool2d_desc;
  stage.resize2d_desc = desc.resize2d_desc;
  stage.softmax_desc = desc.softmax_desc;
  stage.topk_desc = desc.topk_desc;
  stage.inputs = inputs;
  stage.outputs = outputs;
  stage.output_descs = output_descs;

  const auto manifest_dispatch =
      gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(
          desc.stage_manifest.custom_kernel);
  if (manifest_dispatch.valid) {
    stage.dispatch_kernel_family = manifest_dispatch.kernel_family;
    stage.dispatch_entry_point = manifest_dispatch.entry_point;
    stage.dispatch_kernel_family_id = manifest_dispatch.kernel_family_id;
    stage.dispatch_flags = manifest_dispatch.flags;
    stage.dispatch_threads_per_threadgroup =
        manifest_dispatch.threads_per_threadgroup;
    stage.dispatch_precompiled_kernel_required =
        manifest_dispatch.precompiled_binary_required;
    stage.kernel_name = manifest_dispatch.entry_point;
  }

  if (desc.kind == GfxMpsrtStageKind::MSLDispatch) {
    stage.kernel_argument_roles = materialize_gfx_kernel_external_buffer_roles(
        desc.stage_manifest.custom_kernel.external_buffer_abi);
    stage.msl_dispatch_desc = gfx_mpsrt_make_msl_dispatch_desc(
        desc, static_cast<uint32_t>(inputs.size()),
        static_cast<uint32_t>(outputs.size()));
    stage.kernel_buffer_order = gfx_mpsrt_kernel_buffer_order_from_kernel_abi(
        desc.stage_manifest.custom_kernel.external_buffer_abi, stage.inputs,
        stage.outputs);
  }
  return stage;
}

MpsrtModel
build_mpsrt_model_from_builder_plan_or_throw(const GfxMpsrtBuilderPlan &plan) {
  MpsrtModel model;
  std::string error;
  if (!build_mpsrt_model_from_builder_plan(plan, model, &error)) {
    OPENVINO_THROW(error);
  }
  return model;
}

const MpsrtRuntimeResource *
find_mpsrt_external_resource(const MpsrtModel &model,
                             const MpsrtExternalBufferBinding &binding) {
  if (binding.resource_index >= model.resources.size()) {
    return nullptr;
  }
  const auto &resource = model.resources[binding.resource_index];
  if (resource.resource_index != binding.resource_index ||
      resource.arg_index != binding.arg_index ||
      resource.lifetime != MpsrtRuntimeResourceLifetime::External) {
    return nullptr;
  }
  return &resource;
}

const MpsrtRuntimeResource *
find_mpsrt_resource_for_value(const MpsrtModel &model, GfxMpsrtValue value) {
  for (const auto &resource : model.resources) {
    if (resource.has_tensor_value && resource.value == value) {
      return &resource;
    }
  }
  return nullptr;
}

const MpsrtRuntimeTensor *find_mpsrt_tensor(const MpsrtModel &model,
                                            GfxMpsrtValue value) {
  for (const auto &tensor : model.tensors) {
    if (tensor.value == value) {
      return &tensor;
    }
  }
  return nullptr;
}

const GfxMpsrtStorageBridgeDesc *
find_mpsrt_storage_bridge(const MpsrtModel &model, GfxMpsrtValue value) {
  for (const auto &bridge : model.storage_bridges) {
    if (bridge.value == value) {
      return &bridge;
    }
  }
  return nullptr;
}

bool mpsrt_value_list_contains(const std::vector<GfxMpsrtValue> &values,
                               GfxMpsrtValue value) {
  return has_value(values, value);
}

size_t mpsrt_model_external_buffer_abi_count(const MpsrtModel &model) {
  size_t count = model.external_buffer_roles.size();
  if (model.external_buffer_bindings.empty()) {
    return std::max(count, model.external_values.size());
  }
  for (const auto &binding : model.external_buffer_bindings) {
    count = std::max(count, static_cast<size_t>(binding.arg_index) + 1u);
  }
  return count;
}

size_t
mpsrt_model_resource_lifetime_count(const MpsrtModel &model,
                                    MpsrtRuntimeResourceLifetime lifetime) {
  return static_cast<size_t>(std::count_if(
      model.resources.begin(), model.resources.end(),
      [&](const auto &resource) { return resource.lifetime == lifetime; }));
}

GfxMpsrtStorageBridgeDirection mpsrt_model_external_bridge_direction_for_value(
    const MpsrtModel &model, GfxMpsrtValue value,
    GfxMpsrtStorageBridgeDirection fallback_direction) {
  if (model.storage_bridges.empty()) {
    return fallback_direction;
  }
  if (const auto *bridge = find_mpsrt_storage_bridge(model, value)) {
    return bridge->direction;
  }
  return GfxMpsrtStorageBridgeDirection::Unknown;
}

bool mpsrt_model_tensor_binding_plan(
    const MpsrtModel &model, std::vector<MpsrtTensorBindingPlanEntry> &plan,
    std::string *error) {
  plan.clear();
  std::vector<GfxMpsrtValue> external_values = model.external_values;
  std::vector<GfxMpsrtValue> external_output_values =
      model.external_output_values;
  if (external_output_values.empty()) {
    external_output_values = model.output_values;
  }

  auto append_resource = [&](const MpsrtRuntimeResource &resource,
                             uint32_t arg_index) -> bool {
    MpsrtTensorBindingPlanEntry entry{};
    entry.resource_index = resource.resource_index;
    entry.lifetime = resource.lifetime;
    entry.arg_index = arg_index;
    entry.role = resource.role;
    entry.has_tensor_value = resource.has_tensor_value;
    entry.value = resource.value;
    entry.tensor_desc = resource.tensor_desc;
    if (resource.has_tensor_value) {
      entry.bridge_direction = external_binding_bridge_direction(
          model, resource.value,
          has_value(external_output_values, resource.value));
    }
    plan.push_back(entry);
    return true;
  };

  if (!model.external_buffer_bindings.empty()) {
    plan.reserve(model.external_buffer_bindings.size() +
                 model.resources.size());
    for (const auto &external : model.external_buffer_bindings) {
      const auto *resource = find_mpsrt_external_resource(model, external);
      if (!resource) {
        return fail(
            error,
            "GFX MPSRT: external binding references an invalid resource");
      }
      if (!append_resource(*resource, external.arg_index)) {
        return false;
      }
    }
  } else {
    plan.reserve(external_values.size() + model.resources.size());
    for (size_t i = 0; i < external_values.size(); ++i) {
      const auto *resource = find_resource_for_value_and_lifetime(
          model, external_values[i], MpsrtRuntimeResourceLifetime::External);
      if (!resource) {
        return fail(
            error,
            "GFX MPSRT: external value has no external runtime resource");
      }
      if (!append_resource(*resource, static_cast<uint32_t>(i))) {
        return false;
      }
    }
  }

  for (const auto &resource : model.resources) {
    if (resource.lifetime != MpsrtRuntimeResourceLifetime::Model &&
        resource.lifetime != MpsrtRuntimeResourceLifetime::Transient) {
      continue;
    }
    if (!resource.has_tensor_value) {
      return fail(error, "GFX MPSRT: owned runtime resource is not a tensor");
    }
    if (!append_resource(resource, resource.arg_index)) {
      return false;
    }
  }
  return true;
}

bool adapt_mpsrt_model_to_external_buffer_abi(MpsrtModel &model,
                                              uint32_t arg_count,
                                              uint32_t output_arg_count,
                                              std::string *error) {
  return materialize_multi_stage_external_buffer_abi(model, arg_count,
                                                     output_arg_count, error);
}

} // namespace mpsrt
} // namespace gfx_plugin
} // namespace ov
