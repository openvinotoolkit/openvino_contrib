// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_binding.hpp"

#include <cstdint>
#include <memory>

#include "mlir/codegen_common.hpp"
#include "mlir/msl_codegen_apple_msl_common.hpp"
#include "mlir/msl_codegen_apple_msl_op_kinds.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/elu.hpp"

namespace ov {
namespace gfx_plugin {

std::optional<KernelSource> make_apple_metal_unary_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return std::nullopt;
  }

  const auto activation = unary_activation_kind_from_node(*node);
  if (!activation) {
    return std::nullopt;
  }

  UnaryCodegenDesc desc{};
  desc.element_type = node->get_output_element_type(0);
  desc.activation = *activation;
  desc.alpha = 0.0f;
  if (auto elu = ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
    desc.alpha = static_cast<float>(elu->get_alpha());
  }
  if (auto clamp = ov::as_type_ptr<const ov::op::v0::Clamp>(node)) {
    desc.clamp_min = clamp->get_min();
    desc.clamp_max = clamp->get_max();
  }

  source.entry_point = "unary_kernel";
  source.msl_generator = [desc](mlir::ModuleOp module) mutable {
    return generate_msl_from_mlir(module, desc);
  };
  const auto out_shape = output_shape_for_codegen(source.module, node);
  const int32_t num_elements = static_cast<int32_t>(ov::shape_size(out_shape));
  require_apple_msl_generated_kernel_source_binding(
      source, node->get_type_name(), "unary_kernel", {num_elements});
  return source;
}

} // namespace gfx_plugin
} // namespace ov
