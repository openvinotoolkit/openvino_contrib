// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_binding.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/codegen_common.hpp"
#include "mlir/msl_codegen_apple_msl_common.hpp"
#include "mlir/msl_codegen_apple_msl_op_kinds.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/swish.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

std::optional<float> scalar_float_constant_input(const std::shared_ptr<const ov::Node>& node,
                                                 size_t input_idx) {
  if (!node || input_idx >= node->get_input_size()) {
    return std::nullopt;
  }
  const auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(
      node->input_value(input_idx).get_node_shared_ptr());
  if (!constant ||
      constant->get_output_element_type(0) != node->get_input_element_type(0) ||
      !constant->get_output_partial_shape(0).is_static() ||
      ov::shape_size(constant->get_output_shape(0)) != 1) {
    return std::nullopt;
  }
  const auto values = constant->cast_vector<float>();
  if (values.empty()) {
    return std::nullopt;
  }
  return values.front();
}

bool scalar_float_input(const std::shared_ptr<const ov::Node>& node,
                        size_t input_idx) {
  if (!node || input_idx >= node->get_input_size()) {
    return false;
  }
  return node->get_input_element_type(input_idx) == node->get_input_element_type(0) &&
         node->get_input_partial_shape(input_idx).is_static() &&
         ov::shape_size(node->get_input_shape(input_idx)) == 1;
}

bool unary_input_abi_supported(const std::shared_ptr<const ov::Node>& node) {
  if (ov::as_type_ptr<const ov::op::v4::Swish>(node)) {
    return node->get_input_size() == 1 ||
           (node->get_input_size() == 2 && scalar_float_input(node, 1));
  }
  return node->get_input_size() == 1;
}

bool swish_uses_runtime_beta_input(const std::shared_ptr<const ov::Node>& node) {
  return ov::as_type_ptr<const ov::op::v4::Swish>(node) &&
         node->get_input_size() == 2 &&
         scalar_float_input(node, 1) &&
         !scalar_float_constant_input(node, 1).has_value();
}

}  // namespace

std::optional<KernelSource> make_apple_metal_unary_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return std::nullopt;
  }

  const auto activation = unary_activation_kind_from_node(*node);
  if (!activation) {
    return std::nullopt;
  }
  if (!unary_input_abi_supported(node)) {
    return std::nullopt;
  }

  UnaryCodegenDesc desc{};
  desc.element_type = node->get_output_element_type(0);
  desc.activation = *activation;
  desc.swish_beta_runtime_input = swish_uses_runtime_beta_input(node);
  desc.alpha = 0.0f;
  if (auto elu = ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
    desc.alpha = static_cast<float>(elu->get_alpha());
  }
  if (auto clamp = ov::as_type_ptr<const ov::op::v0::Clamp>(node)) {
    desc.clamp_min = clamp->get_min();
    desc.clamp_max = clamp->get_max();
  }
  if (auto gelu = ov::as_type_ptr<const ov::op::v7::Gelu>(node)) {
    desc.gelu_tanh_approximation =
        gelu->get_approximation_mode() == ov::op::GeluApproximationMode::TANH;
  }
  if (auto swish = ov::as_type_ptr<const ov::op::v4::Swish>(node)) {
    desc.alpha = 1.0f;
    if (swish->get_input_size() == 2 && !desc.swish_beta_runtime_input) {
      desc.alpha = *scalar_float_constant_input(node, 1);
    }
  }

  source.entry_point = desc.swish_beta_runtime_input
                           ? "unary_swish_runtime_beta_kernel"
                           : "unary_kernel";
  source.msl_generator = [desc](mlir::ModuleOp module) mutable {
    return generate_msl_from_mlir(module, desc);
  };
  const auto out_shape = output_shape_for_codegen(source.module, node);
  const int32_t num_elements = static_cast<int32_t>(ov::shape_size(out_shape));
  if (desc.swish_beta_runtime_input) {
    auto binding = make_backend_custom_kernel_roles_binding_plan(
        node->get_type_name(),
        source.entry_point,
        {GfxKernelBufferRole::TensorInput,
         GfxKernelBufferRole::TensorInput,
         GfxKernelBufferRole::TensorOutput,
         GfxKernelBufferRole::ScalarParam});
    if (!binding.valid || binding.scalar_arg_count != 1) {
      return std::nullopt;
    }
    binding.runtime_binding.scalar_args = {num_elements};
    binding.stage_manifest.custom_kernel.scalar_args = {num_elements};
    if (!configure_backend_custom_kernel_source_from_binding_plan(source, binding)) {
      return std::nullopt;
    }
  } else {
    require_apple_msl_generated_kernel_source_binding(
        source, node->get_type_name(), "unary_kernel", {num_elements});
  }
  return source;
}

} // namespace gfx_plugin
} // namespace ov
