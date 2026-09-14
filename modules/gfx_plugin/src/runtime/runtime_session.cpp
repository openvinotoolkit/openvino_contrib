// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/runtime_session.hpp"

#include <algorithm>
#include <utility>

#include "openvino/core/except.hpp"
#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

ResourceBindingTable
ResourceBindingTable::for_stage(const std::vector<GpuTensor *> &inputs,
                                const std::vector<GpuTensor *> &outputs) {
  ResourceBindingTable table;
  table.m_inputs = inputs;
  table.m_outputs = outputs;
  return table;
}

namespace {

std::vector<std::string> descriptor_input_region_ids(
    const RuntimeStageExecutableDescriptor &descriptor) {
  std::vector<std::string> ids;
  ids.reserve(descriptor.input_bindings.size());
  for (const auto &binding : descriptor.input_bindings) {
    ids.push_back(binding.memory_region_id);
  }
  return ids;
}

std::vector<std::string> descriptor_output_region_ids(
    const RuntimeStageExecutableDescriptor &descriptor) {
  std::vector<std::string> ids;
  ids.reserve(descriptor.output_bindings.size());
  for (const auto &binding : descriptor.output_bindings) {
    ids.push_back(binding.memory_region_id);
  }
  return ids;
}

bool binding_region_ids_match(
    const std::vector<RuntimeTensorBindingContract> &bindings,
    const std::vector<std::string> &region_ids) noexcept {
  if (region_ids.size() != bindings.size()) {
    return false;
  }
  for (size_t i = 0; i < bindings.size(); ++i) {
    if (region_ids[i] != bindings[i].memory_region_id) {
      return false;
    }
  }
  return true;
}

} // namespace

ResourceBindingTable ResourceBindingTable::for_stage(
    const std::vector<GpuTensor *> &inputs,
    const std::vector<GpuTensor *> &outputs,
    const RuntimeStageExecutableDescriptor &descriptor) {
  auto table = for_stage(inputs, outputs);
  table.m_input_region_ids = descriptor_input_region_ids(descriptor);
  table.m_output_region_ids = descriptor_output_region_ids(descriptor);
  return table;
}

bool ResourceBindingTable::compatible_with(
    const RuntimeStageExecutableDescriptor &descriptor) const noexcept {
  if (descriptor.stage_record_key == 0 || descriptor.kernel_id.empty() ||
      descriptor.stage_name.empty() || descriptor.abi_fingerprint.empty() ||
      descriptor.artifact_key.empty()) {
    return false;
  }
  if (descriptor.payload_kind != KernelArtifactPayloadKind::None &&
      !descriptor.payload) {
    return false;
  }
  if (inputs().size() != descriptor.input_bindings.size() ||
      outputs().size() != descriptor.output_bindings.size()) {
    return false;
  }
  if (!m_input_region_ids.empty() &&
      !binding_region_ids_match(descriptor.input_bindings,
                                m_input_region_ids)) {
    return false;
  }
  if (!m_output_region_ids.empty() &&
      !binding_region_ids_match(descriptor.output_bindings,
                                m_output_region_ids)) {
    return false;
  }
  const auto binding_complete = [](const RuntimeTensorBindingContract &binding) {
    return !binding.logical_name.empty() && !binding.memory_region_id.empty() &&
           !binding.role.empty() && !binding.element_type.empty() &&
           !binding.partial_shape.empty() && !binding.layout.empty() &&
           !binding.storage_kind.empty() && !binding.lifetime_class.empty() &&
           !binding.alias_group.empty();
  };
  if (!std::all_of(descriptor.input_bindings.begin(),
                   descriptor.input_bindings.end(), binding_complete) ||
      !std::all_of(descriptor.output_bindings.begin(),
                   descriptor.output_bindings.end(), binding_complete)) {
    return false;
  }
  return true;
}

PreparedKernelExecutable::PreparedKernelExecutable(
    const RuntimeStageExecutableDescriptor &descriptor)
    : m_descriptor(&descriptor) {}

void PreparedKernelExecutable::prepare(GpuStage &stage,
                                       GpuBufferManager *buffer_manager,
                                       ResourceBindingTable bindings) {
  OPENVINO_ASSERT(m_descriptor,
                  "GFX: prepared kernel executable descriptor is null");
  OPENVINO_ASSERT(bindings.compatible_with(*m_descriptor),
                  "GFX: resource binding table is not compatible with "
                  "compiler-owned runtime descriptor for kernel ",
                  m_descriptor->kernel_id);

  stage.set_inputs(bindings.inputs());
  stage.prepare_runtime_handle(buffer_manager);
  m_bindings = std::move(bindings);
  m_prepared = true;
}

void PreparedKernelExecutable::bind(ResourceBindingTable bindings) {
  OPENVINO_ASSERT(m_descriptor,
                  "GFX: prepared kernel executable descriptor is null");
  OPENVINO_ASSERT(bindings.compatible_with(*m_descriptor),
                  "GFX: resource binding table is not compatible with "
                  "compiler-owned runtime descriptor for kernel ",
                  m_descriptor->kernel_id);
  m_bindings = std::move(bindings);
}

RuntimeSession::RuntimeSession(
    std::shared_ptr<const RuntimeExecutableDescriptor> descriptor)
    : m_descriptor(std::move(descriptor)) {
  OPENVINO_ASSERT(m_descriptor, "GFX: runtime session descriptor is null");
  OPENVINO_ASSERT(!m_descriptor->target_fingerprint.empty(),
                  "GFX: runtime session descriptor target is empty");
}

size_t RuntimeSession::stage_count() const noexcept {
  return m_descriptor ? m_descriptor->stages.size() : 0;
}

ResourceBindingTable RuntimeSession::make_binding_table(
    size_t stage_index, const std::vector<GpuTensor *> &inputs,
    const std::vector<GpuTensor *> &outputs) const {
  return ResourceBindingTable::for_stage(inputs, outputs,
                                         stage_descriptor(stage_index));
}

PreparedKernelExecutable
RuntimeSession::prepare_stage(size_t stage_index, GpuStage &stage,
                              GpuBufferManager *buffer_manager,
                              ResourceBindingTable bindings) const {
  auto prepared = PreparedKernelExecutable(stage_descriptor(stage_index));
  prepared.prepare(stage, buffer_manager, std::move(bindings));
  return prepared;
}

const RuntimeStageExecutableDescriptor &
RuntimeSession::stage_descriptor(size_t stage_index) const {
  OPENVINO_ASSERT(m_descriptor, "GFX: runtime session descriptor is null");
  OPENVINO_ASSERT(stage_index < m_descriptor->stages.size(),
                  "GFX: runtime session stage index ",
                  stage_index,
                  " is out of range for compiler-owned descriptor with ",
                  m_descriptor->stages.size(),
                  " stages");
  const auto &descriptor = m_descriptor->stages[stage_index];
  OPENVINO_ASSERT(descriptor.stage_index == stage_index,
                  "GFX: runtime session descriptor stage index drift at ",
                  stage_index);
  return descriptor;
}

} // namespace gfx_plugin
} // namespace ov
