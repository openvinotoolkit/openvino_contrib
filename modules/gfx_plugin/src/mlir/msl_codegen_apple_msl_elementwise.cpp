// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mlir/msl_codegen_apple_msl_binding.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "mlir/codegen_common.hpp"
#include "mlir/msl_codegen_apple_msl_common.hpp"
#include "mlir/msl_codegen_apple_msl_op_kinds.hpp"
#include "openvino/core/shape_util.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {

std::optional<KernelSource> make_apple_metal_elementwise_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return std::nullopt;
  }

  const std::string type = node->get_type_name();
  if (type == "Select") {
    const auto out_shape = output_shape_for_codegen(source.module, node);
    source.entry_point = "select_kernel";
    source.msl_generator = [element_type = node->get_output_element_type(0)](
                               mlir::ModuleOp module) {
      return generate_msl_for_select(module, element_type);
    };
    const std::vector<int32_t> scalars{
        static_cast<int32_t>(ov::shape_size(out_shape)),
        static_cast<int32_t>(out_shape.empty() ? 1 : out_shape.size())};
    require_apple_msl_generated_kernel_source_binding(source, "Select",
                                                      "select_kernel", scalars);
    return source;
  }

  auto kind = eltwise_kind_from_node(*node);
  if (!kind) {
    return std::nullopt;
  }

  EltwiseCodegenDesc desc{};
  desc.element_type = node->get_output_element_type(0);
  desc.input0_type = node->get_input_element_type(0);
  desc.input1_type = node->get_input_element_type(1);
  desc.output_type = node->get_output_element_type(0);
  desc.eltwise_kind = *kind;
  if (auto input_activation = activation_kind_from_module_attr(
          source.module, "gfx.input_activation_kind")) {
    desc.has_input_activation = true;
    desc.input_activation = *input_activation;
    if (auto input_attr = source.module->getAttrOfType<mlir::IntegerAttr>(
            "gfx.input_activation_input")) {
      desc.input_activation_index =
          static_cast<uint32_t>(std::max<int64_t>(input_attr.getInt(), 0));
    }
    if (auto alpha_attr = source.module->getAttrOfType<mlir::FloatAttr>(
            "gfx.input_activation_alpha")) {
      desc.input_activation_alpha =
          static_cast<float>(alpha_attr.getValueAsDouble());
    }
  }

  const bool dynamic_shape = !node->get_output_partial_shape(0).is_static() ||
                             !node->get_input_partial_shape(0).is_static() ||
                             !node->get_input_partial_shape(1).is_static();
  const auto out_shape = output_shape_for_codegen(source.module, node);
  desc.out_shape = to_i64_shape(out_shape);
  desc.num_elements = static_cast<uint32_t>(ov::shape_size(out_shape));
  const auto input0_shape = shape_from_entry_argument_or_partial(
      source.module, 0, node->get_input_partial_shape(0));
  const auto input1_shape = shape_from_entry_argument_or_partial(
      source.module, 1, node->get_input_partial_shape(1));
  const auto perm0 = read_absorbed_input_permutation(source.module, 0);
  const auto perm1 = read_absorbed_input_permutation(source.module, 1);
  desc.is_broadcast = dynamic_shape || !perm0.empty() || !perm1.empty() ||
                      (input0_shape != input1_shape) ||
                      (input0_shape != out_shape) ||
                      (input1_shape != out_shape);
  if (!perm0.empty()) {
    auto strides = compute_permuted_broadcast_element_strides(
        input0_shape,
        static_shape_or_placeholder(node->get_input_partial_shape(0)), perm0,
        out_shape, "GFX Metal");
    desc.stride0.assign(strides.begin(), strides.end());
  } else {
    fill_broadcast_strides(out_shape, input0_shape, desc.stride0);
  }
  if (!perm1.empty()) {
    auto strides = compute_permuted_broadcast_element_strides(
        input1_shape,
        static_shape_or_placeholder(node->get_input_partial_shape(1)), perm1,
        out_shape, "GFX Metal");
    desc.stride1.assign(strides.begin(), strides.end());
  } else {
    fill_broadcast_strides(out_shape, input1_shape, desc.stride1);
  }

  source.entry_point = "eltwise_kernel";
  source.msl_generator = [desc](mlir::ModuleOp module) mutable {
    return generate_msl_from_mlir(module, desc);
  };
  std::vector<int32_t> scalars = {static_cast<int32_t>(desc.num_elements),
                                  static_cast<int32_t>(out_shape.size())};
  require_apple_msl_generated_kernel_source_binding(source, type,
                                                    "eltwise_kernel", scalars);
  return source;
}

} // namespace gfx_plugin
} // namespace ov
