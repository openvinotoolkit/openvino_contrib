// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_binding.hpp"

#include <memory>
#include <vector>

#include "mlir/codegen_common.hpp"
#include "mlir/msl_codegen_apple_msl_op_kinds.hpp"
#include "openvino/core/shape_util.hpp"

namespace ov {
namespace gfx_plugin {

std::optional<KernelSource> make_apple_metal_reduction_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return std::nullopt;
  }

  const auto reduce_kind = reduce_kind_from_node(*node);
  if (!reduce_kind) {
    return std::nullopt;
  }

  ReduceCodegenDesc desc{};
  desc.element_type = node->get_output_element_type(0);
  desc.kind = *reduce_kind;
  source.entry_point = "reduce_kernel";
  source.msl_generator = [desc](mlir::ModuleOp module) mutable {
    return generate_msl_from_mlir(module, desc);
  };
  const auto input_shape = node->get_input_shape(0);
  const auto output_shape = node->get_output_shape(0);
  require_apple_msl_generated_kernel_source_binding(
      source, node->get_type_name(), "reduce_kernel",
      {static_cast<int32_t>(ov::shape_size(output_shape)),
       static_cast<int32_t>(input_shape.size())});
  return source;
}

} // namespace gfx_plugin
} // namespace ov
