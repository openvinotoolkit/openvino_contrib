// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_eltwise_kernel_unit.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "openvino/core/shape_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "kernel_ir/opencl_kernels/eltwise_compare_select_kernel.hpp"
#include "kernel_ir/opencl_kernels/eltwise_kernel.hpp"
#include "kernel_ir/opencl_kernels/eltwise_logical_bool_kernel.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

bool is_f32_tensor_type(const ov::element::Type &type) {
  return type == ov::element::f32;
}

bool is_f16_tensor_type(const ov::element::Type &type) {
  return type == ov::element::f16;
}

bool is_bool_tensor_type(const ov::element::Type &type) {
  return type == ov::element::boolean;
}

bool is_i32_tensor_type(const ov::element::Type &type) {
  return type == ov::element::i32;
}

bool is_opencl_binary_tensor_type(const ov::element::Type &type) {
  return is_f16_tensor_type(type) || is_f32_tensor_type(type) ||
         is_i32_tensor_type(type);
}

const char *opencl_scalar_type_suffix(const ov::element::Type &type) {
  if (is_f16_tensor_type(type)) {
    return "f16";
  }
  if (is_f32_tensor_type(type)) {
    return "f32";
  }
  if (is_i32_tensor_type(type)) {
    return "i32";
  }
  return "unknown";
}

bool checked_u32(uint64_t value, uint32_t &out) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  out = static_cast<uint32_t>(value);
  return true;
}

uint64_t shape_product_range(const ov::Shape &shape, size_t begin,
                             size_t end) {
  uint64_t product = 1;
  for (size_t axis = begin; axis < end; ++axis) {
    product *= shape[axis];
  }
  return product;
}

bool append_shape_u32(const ov::Shape &shape, size_t max_rank,
                      std::vector<uint32_t> &values) {
  if (shape.size() > max_rank) {
    return false;
  }
  for (const auto dim : shape) {
    uint32_t value = 0;
    if (!checked_u32(dim, value)) {
      return false;
    }
    values.push_back(value);
  }
  values.insert(values.end(), max_rank - shape.size(), 1u);
  return true;
}

bool same_static_shape(const std::shared_ptr<const ov::Node> &node,
                       size_t input_a, size_t input_b) {
  if (!node || input_a >= node->get_input_size() ||
      input_b >= node->get_input_size()) {
    return false;
  }
  if (!node->get_input_partial_shape(input_a).is_static() ||
      !node->get_input_partial_shape(input_b).is_static()) {
    return false;
  }
  return node->get_input_shape(input_a) == node->get_input_shape(input_b);
}

bool is_static_scalar_like_input(const std::shared_ptr<const ov::Node> &node,
                                 size_t input_idx) {
  if (!node || input_idx >= node->get_input_size() ||
      !node->get_input_partial_shape(input_idx).is_static()) {
    return false;
  }
  return ov::shape_size(node->get_input_shape(input_idx)) == 1;
}

bool input_static_element_count_matches_output(
    const std::shared_ptr<const ov::Node> &node, size_t input_idx,
    size_t output_idx) {
  if (!node || input_idx >= node->get_input_size() ||
      output_idx >= node->get_output_size()) {
    return false;
  }
  if (!node->get_input_partial_shape(input_idx).is_static() ||
      !node->get_output_partial_shape(output_idx).is_static()) {
    return false;
  }
  return ov::shape_size(node->get_input_shape(input_idx)) ==
         ov::shape_size(node->get_output_shape(output_idx));
}

std::optional<float>
scalar_f32_constant_input(const std::shared_ptr<const ov::Node> &node,
                          size_t input_idx) {
  if (!node || input_idx >= node->get_input_size()) {
    return std::nullopt;
  }
  auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(
      node->input_value(input_idx).get_node_shared_ptr());
  if (!constant || !is_f32_tensor_type(constant->get_output_element_type(0)) ||
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

std::optional<uint32_t> aligned_broadcast_stride(
    const ov::Shape &input_shape, const ov::Shape &output_shape,
    size_t output_axis) {
  const size_t output_rank = output_shape.size();
  const size_t input_rank = input_shape.size();
  if (output_axis >= output_rank || input_rank > output_rank) {
    return std::nullopt;
  }
  if (output_axis < output_rank - input_rank) {
    return 0u;
  }
  const size_t input_axis = output_axis - (output_rank - input_rank);
  const size_t input_dim = input_shape[input_axis];
  const size_t output_dim = output_shape[output_axis];
  if (input_dim != output_dim && input_dim != 1) {
    return std::nullopt;
  }
  if (input_dim == 1 && output_dim != 1) {
    return 0u;
  }
  uint32_t stride = 0;
  if (!checked_u32(shape_product_range(input_shape, input_axis + 1,
                                       input_rank),
                   stride)) {
    return std::nullopt;
  }
  return stride;
}

bool append_aligned_broadcast_strides_u32(const ov::Shape &input_shape,
                                          const ov::Shape &output_shape,
                                          size_t max_rank,
                                          std::vector<uint32_t> &values) {
  if (output_shape.size() > max_rank ||
      input_shape.size() > output_shape.size()) {
    return false;
  }
  for (size_t axis = 0; axis < output_shape.size(); ++axis) {
    const auto stride =
        aligned_broadcast_stride(input_shape, output_shape, axis);
    if (!stride) {
      return false;
    }
    values.push_back(*stride);
  }
  values.insert(values.end(), max_rank - output_shape.size(), 0u);
  return true;
}

bool output_type_matches(const std::shared_ptr<const ov::Node> &node,
                         const ov::element::Type &output_type) {
  return node && node->get_output_size() == 1 &&
         node->get_output_element_type(0) == output_type;
}

std::optional<std::vector<uint32_t>>
broadcast_static_u32_scalars(const std::shared_ptr<const ov::Node> &node,
                             const std::vector<ov::element::Type> &input_types,
                             const ov::element::Type &output_type) {
  if (!node || node->get_input_size() != input_types.size() ||
      node->get_output_size() != 1 ||
      !node->get_output_partial_shape(0).is_static() ||
      !output_type_matches(node, output_type)) {
    return std::nullopt;
  }

  for (size_t input_idx = 0; input_idx < input_types.size(); ++input_idx) {
    if (!node->get_input_partial_shape(input_idx).is_static() ||
        node->get_input_element_type(input_idx) != input_types[input_idx]) {
      return std::nullopt;
    }
  }

  const auto &output_shape = node->get_output_shape(0);
  const size_t rank = output_shape.size();
  if (rank == 0 || rank > 4 || ov::shape_size(output_shape) == 0) {
    return std::nullopt;
  }

  for (size_t input_idx = 0; input_idx < input_types.size(); ++input_idx) {
    const auto &input_shape = node->get_input_shape(input_idx);
    if (input_shape.size() > rank || ov::shape_size(input_shape) == 0) {
      return std::nullopt;
    }
  }

  for (size_t axis = 0; axis < rank; ++axis) {
    if (output_shape[axis] == 0) {
      return std::nullopt;
    }
    for (size_t input_idx = 0; input_idx < input_types.size(); ++input_idx) {
      if (!aligned_broadcast_stride(node->get_input_shape(input_idx),
                                    output_shape, axis)) {
        return std::nullopt;
      }
    }
  }

  uint32_t total = 0;
  if (!checked_u32(ov::shape_size(output_shape), total)) {
    return std::nullopt;
  }
  (void)total;

  std::vector<uint32_t> scalars;
  scalars.reserve(1 + 4 + 4 * input_types.size());
  scalars.push_back(static_cast<uint32_t>(rank));
  if (!append_shape_u32(output_shape, 4, scalars)) {
    return std::nullopt;
  }
  for (size_t input_idx = 0; input_idx < input_types.size(); ++input_idx) {
    if (!append_aligned_broadcast_strides_u32(node->get_input_shape(input_idx),
                                              output_shape, 4, scalars)) {
      return std::nullopt;
    }
  }
  return scalars;
}

std::optional<std::vector<uint32_t>> binary_broadcast_static_u32_scalars(
    const std::shared_ptr<const ov::Node> &node) {
  if (!node || node->get_input_size() != 2 || node->get_output_size() != 1) {
    return std::nullopt;
  }
  const auto element_type = node->get_output_element_type(0);
  if (!is_opencl_binary_tensor_type(element_type)) {
    return std::nullopt;
  }
  return broadcast_static_u32_scalars(node, {element_type, element_type},
                                      element_type);
}

std::optional<std::vector<uint32_t>> compare_broadcast_static_u32_scalars(
    const std::shared_ptr<const ov::Node> &node) {
  return broadcast_static_u32_scalars(
      node, {ov::element::f32, ov::element::f32}, ov::element::boolean);
}

std::optional<std::vector<uint32_t>> select_broadcast_static_u32_scalars(
    const std::shared_ptr<const ov::Node> &node) {
  return broadcast_static_u32_scalars(
      node, {ov::element::boolean, ov::element::f32, ov::element::f32},
      ov::element::f32);
}

std::optional<std::vector<uint32_t>>
logical_binary_broadcast_static_u32_scalars(
    const std::shared_ptr<const ov::Node> &node) {
  return broadcast_static_u32_scalars(
      node, {ov::element::boolean, ov::element::boolean}, ov::element::boolean);
}

std::optional<size_t> static_rank(const ov::PartialShape &shape) {
  if (shape.rank().is_dynamic()) {
    return std::nullopt;
  }
  return shape.size();
}

bool partial_same_rank_shapes_compatible(
    const std::shared_ptr<const ov::Node> &node,
    const std::vector<size_t> &input_indices, size_t output_idx) {
  if (!node || output_idx >= node->get_output_size()) {
    return false;
  }
  const auto output_rank =
      static_rank(node->get_output_partial_shape(output_idx));
  if (!output_rank) {
    return false;
  }
  for (const size_t input_idx : input_indices) {
    if (input_idx >= node->get_input_size()) {
      return false;
    }
    const auto input_rank =
        static_rank(node->get_input_partial_shape(input_idx));
    if (!input_rank || *input_rank != *output_rank) {
      return false;
    }
  }
  return true;
}

bool select_dynamic_f16_supported(const std::shared_ptr<const ov::Node> &node) {
  return node && node->get_type_name() == std::string("Select") &&
         node->get_input_size() == 3 && node->get_output_size() == 1 &&
         !node->get_output_partial_shape(0).is_static() &&
         is_bool_tensor_type(node->get_input_element_type(0)) &&
         is_f16_tensor_type(node->get_input_element_type(1)) &&
         is_f16_tensor_type(node->get_input_element_type(2)) &&
         is_f16_tensor_type(node->get_output_element_type(0)) &&
         partial_same_rank_shapes_compatible(node, {0, 1, 2}, 0);
}

bool is_numpy_aligned_binary_broadcast(
    const ov::op::util::BinaryElementwiseArithmetic &op) {
  return op.get_autob().m_type == ov::op::AutoBroadcastType::NUMPY;
}

std::optional<GfxOpenClArtifactOp>
binary_op_code(const std::shared_ptr<const ov::Node> &node) {
  if (ov::as_type_ptr<const ov::op::v1::Add>(node)) {
    return GfxOpenClArtifactOp::Add;
  }
  if (ov::as_type_ptr<const ov::op::v1::Subtract>(node)) {
    return GfxOpenClArtifactOp::Subtract;
  }
  if (ov::as_type_ptr<const ov::op::v1::Multiply>(node)) {
    return GfxOpenClArtifactOp::Multiply;
  }
  if (ov::as_type_ptr<const ov::op::v1::Divide>(node)) {
    return GfxOpenClArtifactOp::Divide;
  }
  if (ov::as_type_ptr<const ov::op::v1::Maximum>(node)) {
    return GfxOpenClArtifactOp::Maximum;
  }
  if (ov::as_type_ptr<const ov::op::v1::Minimum>(node)) {
    return GfxOpenClArtifactOp::Minimum;
  }
  if (ov::as_type_ptr<const ov::op::v1::Power>(node)) {
    return GfxOpenClArtifactOp::Power;
  }
  if (ov::as_type_ptr<const ov::op::v0::SquaredDifference>(node)) {
    return GfxOpenClArtifactOp::SquaredDifference;
  }
  if (ov::as_type_ptr<const ov::op::v1::Mod>(node)) {
    return GfxOpenClArtifactOp::Mod;
  }
  if (ov::as_type_ptr<const ov::op::v1::FloorMod>(node)) {
    return GfxOpenClArtifactOp::FloorMod;
  }
  return std::nullopt;
}

std::optional<GfxOpenClArtifactOp> compare_op_code(std::string_view type) {
  if (type == "Equal")
    return GfxOpenClArtifactOp::Equal;
  if (type == "NotEqual")
    return GfxOpenClArtifactOp::NotEqual;
  if (type == "Greater")
    return GfxOpenClArtifactOp::Greater;
  if (type == "GreaterEqual")
    return GfxOpenClArtifactOp::GreaterEqual;
  if (type == "Less")
    return GfxOpenClArtifactOp::Less;
  if (type == "LessEqual")
    return GfxOpenClArtifactOp::LessEqual;
  return std::nullopt;
}

std::optional<GfxOpenClArtifactOp>
logical_unary_op_code(std::string_view type) {
  if (type == "LogicalNot")
    return GfxOpenClArtifactOp::LogicalNot;
  return std::nullopt;
}

std::optional<GfxOpenClArtifactOp>
logical_binary_op_code(std::string_view type) {
  if (type == "LogicalAnd")
    return GfxOpenClArtifactOp::LogicalAnd;
  if (type == "LogicalOr")
    return GfxOpenClArtifactOp::LogicalOr;
  if (type == "LogicalXor" || type == "Xor")
    return GfxOpenClArtifactOp::LogicalXor;
  return std::nullopt;
}

std::optional<GfxOpenClSourceArtifact>
make_opencl_arithmetic_eltwise_artifact(
    const std::shared_ptr<const ov::Node> &node) {
  auto eltwise =
      ov::as_type_ptr<const ov::op::util::BinaryElementwiseArithmetic>(node);
  const auto op = binary_op_code(node);
  if (!eltwise || !op || eltwise->get_input_size() != 2 ||
      eltwise->get_output_size() != 1 ||
      eltwise->get_output_element_type(0) !=
          eltwise->get_input_element_type(0) ||
      eltwise->get_output_element_type(0) !=
          eltwise->get_input_element_type(1) ||
      !is_opencl_binary_tensor_type(eltwise->get_output_element_type(0))) {
    return std::nullopt;
  }

  const auto element_type = eltwise->get_output_element_type(0);
  const bool is_f32 = is_f32_tensor_type(element_type);
  const std::string type_suffix = opencl_scalar_type_suffix(element_type);
  const bool lhs_matches_output =
      input_static_element_count_matches_output(node, 0, 0);
  const bool rhs_matches_output =
      input_static_element_count_matches_output(node, 1, 0);
  const auto lhs_constant = scalar_f32_constant_input(node, 0);
  const auto rhs_constant = scalar_f32_constant_input(node, 1);
  const std::string type = node->get_type_name();
  const auto source_id = [&type_suffix](std::string_view variant) {
    const std::string suffix =
        variant.empty() ? "binary_" + type_suffix : std::string(variant);
    return "opencl/generated/eltwise_" + suffix;
  };

  if (same_static_shape(node, 0, 1) && lhs_matches_output &&
      rhs_matches_output) {
    const std::string entry_point =
        "gfx_opencl_generated_eltwise_binary_" + type_suffix;
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:" + type + ":" + type_suffix + ":same_shape",
        entry_point,
        /*direct_inputs=*/2,
        /*scalar_arg_count=*/2);
    return make_opencl_source_artifact(std::move(manifest), source_id({}),
                                       opencl_generated_eltwise_kernel_source().source,
                                       {GfxOpenClSourceScalarArg::ElementCount,
                                        GfxOpenClSourceScalarArg::OpCode},
                                       {0, 1}, *op);
  }

  if (!is_numpy_aligned_binary_broadcast(*eltwise)) {
    return std::nullopt;
  }

  if (is_f32 && rhs_constant && lhs_matches_output) {
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:" + type + ":f32:rhs_scalar_const",
        "gfx_opencl_generated_eltwise_const_f32",
        /*direct_inputs=*/1,
        /*scalar_arg_count=*/4);
    return make_opencl_source_artifact(
        std::move(manifest), source_id("const_f32"),
        opencl_generated_eltwise_kernel_source().source,
        {GfxOpenClSourceScalarArg::ElementCount,
         GfxOpenClSourceScalarArg::OpCode, GfxOpenClSourceScalarArg::InputMode,
         GfxOpenClSourceScalarArg::ScalarConstantF32},
        {0}, *op, GfxOpenClArtifactInputMode::RhsScalarConstant, *rhs_constant);
  }
  if (is_f32 && lhs_constant && rhs_matches_output) {
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:" + type + ":f32:lhs_scalar_const",
        "gfx_opencl_generated_eltwise_const_f32",
        /*direct_inputs=*/1,
        /*scalar_arg_count=*/4);
    return make_opencl_source_artifact(
        std::move(manifest), source_id("const_f32"),
        opencl_generated_eltwise_kernel_source().source,
        {GfxOpenClSourceScalarArg::ElementCount,
         GfxOpenClSourceScalarArg::OpCode, GfxOpenClSourceScalarArg::InputMode,
         GfxOpenClSourceScalarArg::ScalarConstantF32},
        {1}, *op, GfxOpenClArtifactInputMode::LhsScalarConstant, *lhs_constant);
  }
  if (is_static_scalar_like_input(node, 1) && lhs_matches_output) {
    const std::string entry_point =
        "gfx_opencl_generated_eltwise_scalar_" + type_suffix;
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:" + type + ":" + type_suffix + ":rhs_scalar",
        entry_point,
        /*direct_inputs=*/2,
        /*scalar_arg_count=*/3);
    return make_opencl_source_artifact(
        std::move(manifest), source_id("scalar_" + type_suffix),
        opencl_generated_eltwise_kernel_source().source,
        {GfxOpenClSourceScalarArg::ElementCount,
         GfxOpenClSourceScalarArg::OpCode, GfxOpenClSourceScalarArg::InputMode},
        {0, 1}, *op, GfxOpenClArtifactInputMode::RhsScalar);
  }
  if (is_static_scalar_like_input(node, 0) && rhs_matches_output) {
    const std::string entry_point =
        "gfx_opencl_generated_eltwise_scalar_" + type_suffix;
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:" + type + ":" + type_suffix + ":lhs_scalar",
        entry_point,
        /*direct_inputs=*/2,
        /*scalar_arg_count=*/3);
    return make_opencl_source_artifact(
        std::move(manifest), source_id("scalar_" + type_suffix),
        opencl_generated_eltwise_kernel_source().source,
        {GfxOpenClSourceScalarArg::ElementCount,
         GfxOpenClSourceScalarArg::OpCode, GfxOpenClSourceScalarArg::InputMode},
        {0, 1}, *op, GfxOpenClArtifactInputMode::LhsScalar);
  }

  auto static_u32_scalars = binary_broadcast_static_u32_scalars(node);
  if (static_u32_scalars) {
    const std::string entry_point =
        "gfx_opencl_generated_eltwise_broadcast_" + type_suffix;
    std::vector<GfxOpenClSourceScalarArg> scalar_args = {
        GfxOpenClSourceScalarArg::ElementCount,
        GfxOpenClSourceScalarArg::OpCode};
    scalar_args.insert(scalar_args.end(), static_u32_scalars->size(),
                       GfxOpenClSourceScalarArg::StaticU32);
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:" + type + ":" + type_suffix + ":broadcast",
        entry_point,
        /*direct_inputs=*/2, static_cast<uint32_t>(scalar_args.size()));
    return make_opencl_source_artifact(
        std::move(manifest), source_id("broadcast_" + type_suffix),
        opencl_generated_eltwise_kernel_source().source,
        std::move(scalar_args), {0, 1}, *op, GfxOpenClArtifactInputMode::Direct,
        0.0f, std::move(*static_u32_scalars));
  }
  return std::nullopt;
}

std::optional<GfxOpenClSourceArtifact>
make_opencl_non_arithmetic_eltwise_artifact(
    const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return std::nullopt;
  }
  const std::string type = node->get_type_name();

  if (type == "Select" && select_dynamic_f16_supported(node)) {
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:Select:bool_f16:dynamic_same_shape",
        "gfx_opencl_generated_eltwise_select_f16_dynamic",
        /*direct_inputs=*/3,
        /*scalar_arg_count=*/1);
    return make_opencl_source_artifact(
        std::move(manifest), "opencl/generated/eltwise_select_f16_dynamic",
        opencl_generated_eltwise_select_f16_dynamic_kernel_source().source,
        {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2},
        GfxOpenClArtifactOp::Identity);
  }

  if (auto op = compare_op_code(type)) {
    if (node->get_input_size() != 2 ||
        !is_bool_tensor_type(node->get_output_element_type(0)) ||
        !is_f32_tensor_type(node->get_input_element_type(0)) ||
        !is_f32_tensor_type(node->get_input_element_type(1))) {
      return std::nullopt;
    }
    if (same_static_shape(node, 0, 1) &&
        input_static_element_count_matches_output(node, 0, 0) &&
        input_static_element_count_matches_output(node, 1, 0)) {
      auto manifest = make_opencl_source_manifest(
          GfxKernelStageFamily::Eltwise,
          "opencl:generated:eltwise:" + type + ":f32:same_shape",
          "gfx_opencl_generated_eltwise_compare_f32",
          /*direct_inputs=*/2,
          /*scalar_arg_count=*/2);
      return make_opencl_source_artifact(
          std::move(manifest), "opencl/generated/eltwise_compare_f32",
          opencl_generated_eltwise_compare_f32_kernel_source().source,
          {GfxOpenClSourceScalarArg::ElementCount,
           GfxOpenClSourceScalarArg::OpCode},
          {0, 1}, *op);
    }
    auto static_u32_scalars = compare_broadcast_static_u32_scalars(node);
    if (static_u32_scalars) {
      std::vector<GfxOpenClSourceScalarArg> scalar_args = {
          GfxOpenClSourceScalarArg::ElementCount,
          GfxOpenClSourceScalarArg::OpCode};
      scalar_args.insert(scalar_args.end(), static_u32_scalars->size(),
                         GfxOpenClSourceScalarArg::StaticU32);
      auto manifest = make_opencl_source_manifest(
          GfxKernelStageFamily::Eltwise,
          "opencl:generated:eltwise:" + type + ":f32:broadcast",
          "gfx_opencl_generated_eltwise_compare_broadcast_f32",
          /*direct_inputs=*/2, static_cast<uint32_t>(scalar_args.size()));
      return make_opencl_source_artifact(
          std::move(manifest), "opencl/generated/eltwise_compare_broadcast_f32",
          opencl_generated_eltwise_compare_broadcast_f32_kernel_source().source,
          std::move(scalar_args), {0, 1}, *op,
          GfxOpenClArtifactInputMode::Direct, 0.0f,
          std::move(*static_u32_scalars));
    }
    return std::nullopt;
  }

  if (type == "Select") {
    if (node->get_input_size() != 3 ||
        !is_f32_tensor_type(node->get_output_element_type(0)) ||
        !is_bool_tensor_type(node->get_input_element_type(0)) ||
        !is_f32_tensor_type(node->get_input_element_type(1)) ||
        !is_f32_tensor_type(node->get_input_element_type(2))) {
      return std::nullopt;
    }
    if (input_static_element_count_matches_output(node, 0, 0) &&
        input_static_element_count_matches_output(node, 1, 0) &&
        input_static_element_count_matches_output(node, 2, 0)) {
      auto manifest = make_opencl_source_manifest(
          GfxKernelStageFamily::Eltwise,
          "opencl:generated:eltwise:Select:bool_f32:same_shape",
          "gfx_opencl_generated_eltwise_select_f32",
          /*direct_inputs=*/3,
          /*scalar_arg_count=*/1);
      return make_opencl_source_artifact(
          std::move(manifest), "opencl/generated/eltwise_select_f32",
          opencl_generated_eltwise_select_f32_kernel_source().source,
          {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2},
          GfxOpenClArtifactOp::Identity);
    }
    auto static_u32_scalars = select_broadcast_static_u32_scalars(node);
    if (static_u32_scalars) {
      std::vector<GfxOpenClSourceScalarArg> scalar_args = {
          GfxOpenClSourceScalarArg::ElementCount};
      scalar_args.insert(scalar_args.end(), static_u32_scalars->size(),
                         GfxOpenClSourceScalarArg::StaticU32);
      auto manifest = make_opencl_source_manifest(
          GfxKernelStageFamily::Eltwise,
          "opencl:generated:eltwise:Select:bool_f32:broadcast",
          "gfx_opencl_generated_eltwise_select_broadcast_f32",
          /*direct_inputs=*/3, static_cast<uint32_t>(scalar_args.size()));
      return make_opencl_source_artifact(
          std::move(manifest), "opencl/generated/eltwise_select_broadcast_f32",
          opencl_generated_eltwise_select_broadcast_f32_kernel_source().source,
          std::move(scalar_args), {0, 1, 2}, GfxOpenClArtifactOp::Identity,
          GfxOpenClArtifactInputMode::Direct, 0.0f,
          std::move(*static_u32_scalars));
    }
    return std::nullopt;
  }

  if (auto op = logical_unary_op_code(type)) {
    if (node->get_input_size() != 1 ||
        !is_bool_tensor_type(node->get_output_element_type(0)) ||
        !is_bool_tensor_type(node->get_input_element_type(0)) ||
        !input_static_element_count_matches_output(node, 0, 0)) {
      return std::nullopt;
    }
    auto manifest = make_opencl_source_manifest(
        GfxKernelStageFamily::Eltwise,
        "opencl:generated:eltwise:" + type + ":bool:same_shape",
        "gfx_opencl_generated_eltwise_logical_unary_bool",
        /*direct_inputs=*/1,
        /*scalar_arg_count=*/2);
    return make_opencl_source_artifact(
        std::move(manifest), "opencl/generated/eltwise_logical_unary_bool",
        opencl_generated_eltwise_logical_unary_bool_kernel_source().source,
        {GfxOpenClSourceScalarArg::ElementCount,
         GfxOpenClSourceScalarArg::OpCode},
        {0}, *op);
  }

  if (auto op = logical_binary_op_code(type)) {
    if (node->get_input_size() != 2 ||
        !is_bool_tensor_type(node->get_output_element_type(0)) ||
        !is_bool_tensor_type(node->get_input_element_type(0)) ||
        !is_bool_tensor_type(node->get_input_element_type(1))) {
      return std::nullopt;
    }
    if (same_static_shape(node, 0, 1) &&
        input_static_element_count_matches_output(node, 0, 0) &&
        input_static_element_count_matches_output(node, 1, 0)) {
      auto manifest = make_opencl_source_manifest(
          GfxKernelStageFamily::Eltwise,
          "opencl:generated:eltwise:" + type + ":bool:same_shape",
          "gfx_opencl_generated_eltwise_logical_binary_bool",
          /*direct_inputs=*/2,
          /*scalar_arg_count=*/2);
      return make_opencl_source_artifact(
          std::move(manifest), "opencl/generated/eltwise_logical_binary_bool",
          opencl_generated_eltwise_logical_binary_bool_kernel_source().source,
          {GfxOpenClSourceScalarArg::ElementCount,
           GfxOpenClSourceScalarArg::OpCode},
          {0, 1}, *op);
    }
    auto static_u32_scalars = logical_binary_broadcast_static_u32_scalars(node);
    if (static_u32_scalars) {
      std::vector<GfxOpenClSourceScalarArg> scalar_args = {
          GfxOpenClSourceScalarArg::ElementCount,
          GfxOpenClSourceScalarArg::OpCode};
      scalar_args.insert(scalar_args.end(), static_u32_scalars->size(),
                         GfxOpenClSourceScalarArg::StaticU32);
      auto manifest = make_opencl_source_manifest(
          GfxKernelStageFamily::Eltwise,
          "opencl:generated:eltwise:" + type + ":bool:broadcast",
          "gfx_opencl_generated_eltwise_logical_binary_broadcast_bool",
          /*direct_inputs=*/2, static_cast<uint32_t>(scalar_args.size()));
      return make_opencl_source_artifact(
          std::move(manifest),
          "opencl/generated/eltwise_logical_binary_broadcast_bool",
          opencl_generated_eltwise_logical_binary_broadcast_bool_kernel_source().source,
          std::move(scalar_args), {0, 1}, *op,
          GfxOpenClArtifactInputMode::Direct, 0.0f,
          std::move(*static_u32_scalars));
    }
    return std::nullopt;
  }

  return std::nullopt;
}

} // namespace

std::optional<GfxOpenClSourceArtifact>
make_opencl_eltwise_source_artifact(
    const std::shared_ptr<const ov::Node> &node,
    std::string_view expected_source_id) {
  auto artifact = make_opencl_arithmetic_eltwise_artifact(node);
  if (!artifact || !artifact->valid) {
    artifact = make_opencl_non_arithmetic_eltwise_artifact(node);
  }
  if (!artifact || !artifact->valid) {
    return std::nullopt;
  }
  if (!expected_source_id.empty() &&
      artifact->artifact_ref.source_id != expected_source_id) {
    return std::nullopt;
  }
  return artifact;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
