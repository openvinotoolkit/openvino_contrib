// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/metal_operation_support.hpp"

#include <algorithm>
#include <exception>
#include <vector>

#include "backends/metal/compiler/apple_vendor_descriptors.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_activation.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_eltwise.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_reduction.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_softmax.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "runtime/gfx_logger.hpp"
#include "transforms/gfx_llm_ops.hpp"

namespace ov {
namespace gfx_plugin {

bool metal_supports_node(const std::shared_ptr<const ov::Node> &node);

namespace compiler {
namespace {

bool is_f32_or_f16(const ov::element::Type &element_type) {
  return element_type == ov::element::f32 || element_type == ov::element::f16;
}

bool all_inputs_static(const std::shared_ptr<const ov::Node> &node) {
  for (size_t i = 0; i < node->get_input_size(); ++i) {
    if (!node->get_input_partial_shape(i).is_static()) {
      return false;
    }
  }
  return true;
}

bool all_outputs_static(const std::shared_ptr<const ov::Node> &node) {
  for (size_t i = 0; i < node->get_output_size(); ++i) {
    if (!node->get_output_partial_shape(i).is_static()) {
      return false;
    }
  }
  return true;
}

bool constant_input(const std::shared_ptr<const ov::Node> &node, size_t index) {
  return index < node->get_input_size() &&
         ov::as_type_ptr<const ov::op::v0::Constant>(
             node->input_value(index).get_node_shared_ptr());
}

bool supports_static_concat(const std::shared_ptr<const ov::Node> &node) {
  return ov::as_type_ptr<const ov::op::v0::Concat>(node) &&
         all_inputs_static(node) &&
         node->get_output_partial_shape(0).is_static();
}

bool supports_static_split(const std::shared_ptr<const ov::Node> &node) {
  if (!node ||
      (!ov::as_type_ptr<const ov::op::v1::Split>(node) &&
       !ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node)) ||
      !node->get_input_partial_shape(0).is_static() ||
      !all_outputs_static(node) || !constant_input(node, 1)) {
    return false;
  }
  if (ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node) &&
      !constant_input(node, 2)) {
    return false;
  }
  return true;
}

bool supports_static_slice(const std::shared_ptr<const ov::Node> &node) {
  if (!node ||
      (!ov::as_type_ptr<const ov::op::v8::Slice>(node) &&
       !ov::as_type_ptr<const ov::op::v1::StridedSlice>(node)) ||
      !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static()) {
    return false;
  }
  const auto input_rank = node->get_input_partial_shape(0).rank();
  const auto output_rank = node->get_output_partial_shape(0).rank();
  if (!input_rank.is_static() || !output_rank.is_static() ||
      input_rank.get_length() != output_rank.get_length()) {
    return false;
  }
  if (auto slice = ov::as_type_ptr<const ov::op::v8::Slice>(node)) {
    return constant_input(slice, 1) && constant_input(slice, 2) &&
           constant_input(slice, 3) &&
           (slice->get_input_size() <= 4 || constant_input(slice, 4));
  }
  auto strided_slice = ov::as_type_ptr<const ov::op::v1::StridedSlice>(node);
  return strided_slice &&
         std::all_of(strided_slice->get_new_axis_mask().begin(),
                     strided_slice->get_new_axis_mask().end(),
                     [](int64_t value) { return value == 0; }) &&
         std::all_of(strided_slice->get_shrink_axis_mask().begin(),
                     strided_slice->get_shrink_axis_mask().end(),
                     [](int64_t value) { return value == 0; }) &&
         std::all_of(strided_slice->get_ellipsis_mask().begin(),
                     strided_slice->get_ellipsis_mask().end(),
                     [](int64_t value) { return value == 0; }) &&
         constant_input(strided_slice, 1) && constant_input(strided_slice, 2) &&
         (strided_slice->get_input_size() <= 3 ||
          constant_input(strided_slice, 3));
}

bool supports_static_f32_transpose_generated_msl(
    const std::shared_ptr<const ov::Node> &node) {
  auto transpose = ov::as_type_ptr<const ov::op::v1::Transpose>(node);
  if (!transpose || transpose->get_input_size() != 2 ||
      transpose->get_input_element_type(0) != ov::element::f32 ||
      transpose->get_output_element_type(0) != ov::element::f32 ||
      !transpose->get_input_partial_shape(0).is_static() ||
      !transpose->get_output_partial_shape(0).is_static() ||
      !constant_input(transpose, 1)) {
    return false;
  }
  const auto &input_shape = transpose->get_input_shape(0);
  const auto &output_shape = transpose->get_output_shape(0);
  if (input_shape.empty() || input_shape.size() != output_shape.size() ||
      ov::shape_size(input_shape) != ov::shape_size(output_shape)) {
    return false;
  }
  const auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(
      transpose->input_value(1).get_node_shared_ptr());
  const auto perm = perm_const->cast_vector<int64_t>();
  if (perm.size() != input_shape.size()) {
    return false;
  }
  std::vector<bool> seen(perm.size(), false);
  for (const auto axis : perm) {
    if (axis < 0 || static_cast<size_t>(axis) >= perm.size() ||
        seen[static_cast<size_t>(axis)]) {
      return false;
    }
    seen[static_cast<size_t>(axis)] = true;
  }
  return true;
}

bool supports_causal_sdpa_generated_msl(
    const std::shared_ptr<const ov::Node> &node) {
  return ov::as_type_ptr<const ov::gfx_plugin::op::GfxSDPAWithCausalMask>(
             node) &&
         metal_supports_node(node);
}

bool supports_mps_softmax_vendor(const std::shared_ptr<const ov::Node> &node) {
  GfxMpsrtSoftmaxAbiDesc desc{};
  return gfx_apple_make_mps_softmax_desc(node, desc);
}

bool supports_mps_gemm_vendor(const std::shared_ptr<const ov::Node> &node) {
  GfxAppleMpsVendorPrimitiveContract contract{};
  return gfx_apple_make_mps_gemm_contract(node, contract);
}

bool supports_mps_pool2d_vendor(const std::shared_ptr<const ov::Node> &node) {
  GfxMpsrtPool2DAbiDesc desc{};
  if (!gfx_apple_make_mps_pool2d_desc(node, desc)) {
    return false;
  }
  GfxAppleMpsVendorPrimitiveContract contract{};
  return gfx_apple_make_mps_pool2d_contract(node, desc, contract);
}

bool is_pooling_node(const std::shared_ptr<const ov::Node> &node) {
  return ov::as_type_ptr<const ov::op::util::MaxPoolBase>(node) ||
         ov::as_type_ptr<const ov::op::util::AvgPoolBase>(node);
}

bool supports_mps_resize2d_vendor(const std::shared_ptr<const ov::Node> &node) {
  GfxMpsrtResize2DAbiDesc desc{};
  if (!gfx_apple_make_mps_resize2d_desc(node, desc)) {
    return false;
  }
  GfxAppleMpsVendorPrimitiveContract contract{};
  return gfx_apple_make_mps_resize2d_contract(node, desc, contract);
}

bool supports_mpsgraph_sdpa_vendor(
    const std::shared_ptr<const ov::Node> &node) {
  GfxMpsrtSdpaAbiDesc desc{};
  return gfx_apple_make_mps_sdpa_desc(node, desc);
}

bool supports_generated_eltwise_msl(
    const std::shared_ptr<const ov::Node> &node) {
  return make_eltwise_msl_kernel_source_plan(node).valid();
}

bool supports_generated_activation_msl(
    const std::shared_ptr<const ov::Node> &node) {
  return make_activation_msl_kernel_source_plan(node).valid();
}

bool supports_generated_reduction_msl(
    const std::shared_ptr<const ov::Node> &node) {
  return make_reduction_msl_kernel_source_plan(node).valid();
}

bool supports_generated_softmax_msl(
    const std::shared_ptr<const ov::Node> &node) {
  return make_softmax_msl_kernel_source_plan(node).valid();
}

OperationSupportResult
query_metal_operation(const std::shared_ptr<const ov::Node> &node) {
  try {
    if (node && supports_mps_gemm_vendor(node)) {
      return make_supported_operation("mps_vendor_primitive",
                                      LoweringRouteKind::VendorPrimitive, 0.80,
                                      "metal/vendor/mps_gemm");
    }
    if (node && supports_mps_softmax_vendor(node)) {
      return make_supported_operation("mps_vendor_primitive",
                                      LoweringRouteKind::VendorPrimitive, 0.80,
                                      "metal/vendor/mps_softmax");
    }
    if (node && supports_generated_softmax_msl(node)) {
      const auto descriptor = softmax_msl_kernel_descriptor(node);
      return make_supported_operation(
          "generated_msl_source", LoweringRouteKind::GeneratedKernel, 0.55,
          descriptor ? std::string(descriptor->kernel_unit_id)
                     : "metal/generated/softmax_f32");
    }
    if (node && supports_mps_pool2d_vendor(node)) {
      return make_supported_operation("mps_vendor_primitive",
                                      LoweringRouteKind::VendorPrimitive, 0.80,
                                      "metal/vendor/mps_pool2d");
    }
    if (node && is_pooling_node(node)) {
      return make_unsupported_operation(
          "missing_apple_pooling_mps_family_route");
    }
    if (node && supports_mps_resize2d_vendor(node)) {
      return make_supported_operation("mps_vendor_primitive",
                                      LoweringRouteKind::VendorPrimitive, 0.80,
                                      "metal/vendor/mps_resize2d");
    }
    if (node && supports_mpsgraph_sdpa_vendor(node)) {
      return make_supported_operation("mpsgraph_vendor_primitive",
                                      LoweringRouteKind::VendorPrimitive, 0.80,
                                      "metal/vendor/mpsgraph_sdpa");
    }
    if (node &&
        (ov::as_type_ptr<const ov::op::v0::ShapeOf>(node) ||
         ov::as_type_ptr<const ov::op::v3::ShapeOf>(node)) &&
        node->get_input_partial_shape(0).rank().is_static()) {
      return make_supported_operation("generated_msl_source",
                                      LoweringRouteKind::GeneratedKernel, 0.55,
                                      "metal/generated/shapeof");
    }
    if (node && ov::as_type_ptr<const ov::op::v4::Range>(node) &&
        node->get_output_partial_shape(0).is_static()) {
      return make_supported_operation("generated_msl_source",
                                      LoweringRouteKind::GeneratedKernel, 0.55,
                                      "metal/generated/range");
    }
    if (node && ov::as_type_ptr<const ov::op::v0::Tile>(node) &&
        is_f32_or_f16(node->get_output_element_type(0)) &&
        node->get_input_partial_shape(0).is_static() &&
        node->get_output_partial_shape(0).is_static()) {
      return make_supported_operation("generated_msl_source",
                                      LoweringRouteKind::GeneratedKernel, 0.55,
                                      "metal/generated/tile");
    }
    if (node && supports_static_concat(node)) {
      return make_supported_operation("generated_msl_source",
                                      LoweringRouteKind::GeneratedKernel, 0.55,
                                      "metal/generated/concat");
    }
    if (node && supports_static_split(node)) {
      return make_supported_operation("generated_msl_source",
                                      LoweringRouteKind::GeneratedKernel, 0.55,
                                      "metal/generated/split");
    }
    if (node && supports_static_slice(node)) {
      return make_supported_operation("generated_msl_source",
                                      LoweringRouteKind::GeneratedKernel, 0.55,
                                      "metal/generated/slice");
    }
    if (node && supports_static_f32_transpose_generated_msl(node)) {
      return make_supported_operation("generated_msl_source",
                                      LoweringRouteKind::GeneratedKernel, 0.55,
                                      "metal/generated/transpose_f32");
    }
    if (node && ov::as_type_ptr<const ov::op::v1::Transpose>(node)) {
      return make_unsupported_operation("missing_metal_transpose_kernel_unit");
    }
    if (node && supports_causal_sdpa_generated_msl(node)) {
      return make_supported_operation("generated_msl_source",
                                      LoweringRouteKind::GeneratedKernel, 0.60,
                                      "metal/generated/sdpa_causal_mask");
    }
    if (node && supports_generated_activation_msl(node)) {
      return make_supported_operation("generated_msl_source",
                                      LoweringRouteKind::GeneratedKernel, 0.55,
                                      "metal/generated/activation");
    }
    if (node && supports_generated_eltwise_msl(node)) {
      return make_supported_operation("generated_msl_source",
                                      LoweringRouteKind::GeneratedKernel, 0.55,
                                      "metal/generated/eltwise");
    }
    if (node && supports_generated_reduction_msl(node)) {
      const auto kind = reduction_kind_from_node(node);
      return make_supported_operation(
          "generated_msl_source", LoweringRouteKind::GeneratedKernel, 0.55,
          kind ? std::string(reduction_msl_kernel_unit_id(*kind))
               : "metal/generated/reduction_f32");
    }
    if (node && metal_supports_node(node)) {
      return make_unsupported_operation("missing_metal_explicit_kernel_unit");
    }
    return make_unsupported_operation("unsupported_by_metal_capabilities");
  } catch (const std::exception &e) {
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("Compiler")
          << "Exception probing node " << node->get_friendly_name() << " ("
          << node->get_type_name() << "): " << e.what();
    }
    return make_unsupported_operation(e.what());
  } catch (...) {
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("Compiler")
          << "Unknown exception probing node " << node->get_friendly_name()
          << " (" << node->get_type_name() << ")";
    }
    return make_unsupported_operation("unknown_probe_exception");
  }
}

class MetalOperationSupportPolicy final : public OperationSupportPolicy {
public:
  OperationSupportResult
  query_operation(const OperationSupportQuery &query) const override {
    return query_metal_operation(query.node);
  }
};

} // namespace

std::shared_ptr<const OperationSupportPolicy>
make_metal_operation_support_policy() {
  static const auto policy = std::make_shared<MetalOperationSupportPolicy>();
  return policy;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
