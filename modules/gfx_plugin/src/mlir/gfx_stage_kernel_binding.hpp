// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "compiler/stage_placement.hpp"
#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "openvino/core/except.hpp"
#include "runtime/executable_descriptor.hpp"

namespace ov {
namespace gfx_plugin {

inline KernelRuntimeBindingState make_stage_direct_kernel_runtime_binding(
    std::vector<size_t> kernel_inputs, size_t input_arg_count,
    std::vector<int32_t> operand_kinds,
    std::vector<int32_t> operand_arg_indices,
    std::vector<int32_t> scalar_args = {}) {
  OPENVINO_ASSERT(
      operand_kinds.size() == operand_arg_indices.size(),
      "GFX MLIR: direct stage runtime binding operand metadata size mismatch");
  return make_kernel_runtime_binding_state(
      std::move(kernel_inputs), input_arg_count, std::move(operand_kinds),
      std::move(operand_arg_indices), std::move(scalar_args));
}

inline KernelRuntimeBindingState
make_stage_compact_buffer_kernel_runtime_binding(size_t input_arg_count) {
  std::vector<size_t> inputs;
  inputs.reserve(input_arg_count);
  for (size_t input_index = 0; input_index < input_arg_count; ++input_index) {
    inputs.push_back(input_index);
  }
  return make_stage_direct_kernel_runtime_binding(std::move(inputs),
                                                  input_arg_count, {}, {});
}

inline KernelRuntimeBindingState make_stage_descriptor_kernel_runtime_binding(
    const RuntimeStageExecutableDescriptor &descriptor,
    std::string_view stage_name) {
  const std::string_view label =
      stage_name.empty() ? std::string_view(descriptor.stage_name) : stage_name;
  const auto &launch_plan = descriptor.launch_plan;
  OPENVINO_ASSERT(launch_plan.valid && !launch_plan.buffer_roles.empty(),
                  "GFX MLIR: descriptor-owned launch plan is missing for stage ",
                  label);
  OPENVINO_ASSERT(
      launch_plan.operand_kinds.size() == launch_plan.operand_arg_indices.size(),
      "GFX MLIR: descriptor-owned operand metadata size mismatch for stage ",
      label);

  const auto roles = materialize_descriptor_launch_roles(launch_plan, label);
  if (descriptor.abi_arg_count != 0) {
    OPENVINO_ASSERT(roles.size() == descriptor.abi_arg_count,
                    "GFX MLIR: descriptor launch-plan ABI count drift for stage ",
                    label);
  }

  const size_t tensor_input_count = static_cast<size_t>(std::count(
      roles.begin(), roles.end(), GfxKernelBufferRole::TensorInput));
  const size_t logical_input_arg_count = static_cast<size_t>(std::count_if(
      roles.begin(), roles.end(), [](GfxKernelBufferRole role) {
        return is_gfx_kernel_buffer_role(role) && !is_gfx_kernel_output_role(role);
      }));
  const size_t scalar_role_count = static_cast<size_t>(std::count(
      roles.begin(), roles.end(), GfxKernelBufferRole::ScalarParam));
  if (!launch_plan.scalar_args.empty()) {
    OPENVINO_ASSERT(launch_plan.scalar_args.size() == scalar_role_count,
                    "GFX MLIR: descriptor scalar ABI count drift for stage ",
                    label);
  }

  const auto &input_mapping = !launch_plan.input_indices.empty()
                                  ? launch_plan.input_indices
                                  : launch_plan.direct_input_indices;
  if (!input_mapping.empty()) {
    OPENVINO_ASSERT(input_mapping.size() == tensor_input_count,
                    "GFX MLIR: descriptor tensor input mapping count drift for "
                    "stage ",
                    label);
  }

  std::vector<size_t> mapped_inputs;
  if (!input_mapping.empty()) {
    mapped_inputs = input_mapping;
  } else {
    mapped_inputs.reserve(tensor_input_count);
    for (size_t input_idx = 0; input_idx < tensor_input_count; ++input_idx) {
      mapped_inputs.push_back(input_idx);
    }
  }

  if (!launch_plan.operand_kinds.empty()) {
    KernelRuntimeBindingState binding;
    binding.inputs = std::move(mapped_inputs);
    binding.input_arg_count = launch_plan.input_arg_count != 0
                                  ? launch_plan.input_arg_count
                                  : logical_input_arg_count;
    binding.scalar_args = launch_plan.scalar_args;
    binding.runtime_param_i64_metadata = descriptor.runtime_param_i64_metadata;
    binding.runtime_param_reduce_keep_dims =
        descriptor.runtime_param_reduce_keep_dims;
    binding.runtime_param_reduce_keep_dims_valid =
        descriptor.runtime_param_reduce_keep_dims_valid;
    binding.operand_kinds = launch_plan.operand_kinds;
    binding.operand_arg_indices = launch_plan.operand_arg_indices;
    return binding;
  }

  GfxKernelStageManifest manifest;
  manifest.valid = true;
  manifest.execution_kind = GfxKernelExecutionKind::CustomKernel;
  manifest.storage = GfxKernelStorageKind::Buffer;
  manifest.custom_kernel.valid = true;
  manifest.custom_kernel.entry_point = std::string(descriptor.entry_point);
  manifest.custom_kernel.external_buffer_abi = make_gfx_kernel_roles_abi(roles);
  const auto manifest_plan =
      make_kernel_runtime_binding_plan_from_stage_manifest(manifest);
  OPENVINO_ASSERT(manifest_plan.valid,
                  "GFX MLIR: descriptor launch-plan roles cannot materialize "
                  "runtime binding for stage ",
                  label);

  auto manifest_binding = manifest_plan.runtime_binding;
  manifest_binding.inputs = std::move(mapped_inputs);
  manifest_binding.scalar_args = launch_plan.scalar_args;
  manifest_binding.runtime_param_i64_metadata =
      descriptor.runtime_param_i64_metadata;
  manifest_binding.runtime_param_reduce_keep_dims =
      descriptor.runtime_param_reduce_keep_dims;
  manifest_binding.runtime_param_reduce_keep_dims_valid =
      descriptor.runtime_param_reduce_keep_dims_valid;
  return manifest_binding;
}

inline KernelRuntimeBindingState
require_stage_backend_custom_kernel_runtime_binding(
    GfxKernelBackendDomain backend_domain, std::string_view stage_type,
    std::string_view entry_point, const std::vector<int32_t> &scalar_args,
    std::string_view stage_name) {
  return require_backend_custom_kernel_runtime_binding(
      backend_domain, stage_type, entry_point, scalar_args, stage_name);
}

inline GfxKernelBackendDomain
stage_custom_kernel_backend_domain(GfxStageBackendDomain stage_domain) {
  switch (stage_domain) {
  case GfxStageBackendDomain::AppleMsl:
    return GfxKernelBackendDomain::AppleMsl;
  case GfxStageBackendDomain::OpenCl:
    return GfxKernelBackendDomain::OpenCl;
  case GfxStageBackendDomain::AppleMps:
  case GfxStageBackendDomain::Unknown:
  default:
    return GfxKernelBackendDomain::Unknown;
  }
}

inline GfxKernelRuntimeBindingPlan
annotate_stage_backend_custom_kernel_module_binding(
    mlir::ModuleOp module, GfxStageBackendDomain stage_domain,
    std::string_view stage_type, std::string_view entry_point,
    std::string_view stage_name) {
  const auto kernel_domain = stage_custom_kernel_backend_domain(stage_domain);
  OPENVINO_ASSERT(kernel_domain != GfxKernelBackendDomain::Unknown,
                  "GFX MLIR: stage ", stage_name,
                  " has no custom-kernel backend domain for ", stage_type);
  auto binding = make_backend_custom_kernel_binding_plan(
      kernel_domain, stage_type, entry_point);
  OPENVINO_ASSERT(binding.valid, "GFX MLIR: ", stage_type, " / ", entry_point,
                  " custom-kernel binding manifest is invalid for stage ",
                  stage_name);
  OPENVINO_ASSERT(
      annotate_backend_custom_kernel_module_with_binding_plan(module, binding),
      "GFX MLIR: ", stage_type, " / ", entry_point,
      " failed to annotate stage custom-kernel binding for stage ",
      stage_name);
  return binding;
}

} // namespace gfx_plugin
} // namespace ov
