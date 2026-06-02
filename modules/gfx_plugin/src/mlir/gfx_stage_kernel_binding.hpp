// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "openvino/core/except.hpp"

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

inline KernelRuntimeBindingState
require_stage_backend_custom_kernel_runtime_binding(
    GfxKernelBackendDomain backend_domain, std::string_view stage_type,
    std::string_view entry_point, const std::vector<int32_t> &scalar_args,
    std::string_view stage_name) {
  return require_backend_custom_kernel_runtime_binding(
      backend_domain, stage_type, entry_point, scalar_args, stage_name);
}

} // namespace gfx_plugin
} // namespace ov
