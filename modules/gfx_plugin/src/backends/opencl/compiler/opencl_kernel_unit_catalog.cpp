// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_kernel_unit_catalog.hpp"

#include "backends/opencl/compiler/opencl_activation_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_conv_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_eltwise_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_interpolate_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_pool_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_range_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_reduction_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_shapeof_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_softmax_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_tile_kernel_unit.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

bool is_matmul_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(ov::as_type_ptr<const ov::op::v0::MatMul>(node));
}

bool is_transpose_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Transpose>(node));
}

bool is_concat_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(ov::as_type_ptr<const ov::op::v0::Concat>(node));
}

bool is_split_node(const std::shared_ptr<const ov::Node> &node) {
  return ov::as_type_ptr<const ov::op::v1::Split>(node) ||
         ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node);
}

} // namespace

const std::vector<OpenClGeneratedKernelUnitSpec> &
opencl_generated_kernel_unit_specs() {
  static const std::vector<OpenClGeneratedKernelUnitSpec> specs = {
      {"opencl/generated/conv2d_f32", "Convolution"},
      {"opencl/generated/group_conv2d_f32", "GroupConvolution"},
      {"opencl/generated/shapeof_i32", "ShapeOf"},
      {"opencl/generated/shapeof_i64", "ShapeOf"},
      {"opencl/generated/range_f32", "Range"},
      {"opencl/generated/range_f16", "Range"},
      {"opencl/generated/range_i64", "Range"},
      {"opencl/generated/range_i64_unit_dynamic", "Range"},
      {"opencl/generated/tile_f32", "Tile"},
      {"opencl/generated/tile_dynamic_f32", "Tile"},
      {"opencl/generated/tile_f16", "Tile"},
      {"opencl/generated/tile_dynamic_f16", "Tile"},
      {"opencl/generated/activation_f32", "Activation"},
      {"opencl/generated/activation_f16", "Activation"},
      {"opencl/generated/activation_runtime_beta_f32", "Activation"},
      {"opencl/generated/activation_runtime_beta_f16", "Activation"},
      {"opencl/generated/eltwise_binary_f32", "Eltwise"},
      {"opencl/generated/eltwise_scalar_f32", "Eltwise"},
      {"opencl/generated/eltwise_const_f32", "Eltwise"},
      {"opencl/generated/eltwise_broadcast_f32", "Eltwise"},
      {"opencl/generated/eltwise_binary_f16", "Eltwise"},
      {"opencl/generated/eltwise_scalar_f16", "Eltwise"},
      {"opencl/generated/eltwise_broadcast_f16", "Eltwise"},
      {"opencl/generated/eltwise_binary_i32", "Eltwise"},
      {"opencl/generated/eltwise_scalar_i32", "Eltwise"},
      {"opencl/generated/eltwise_broadcast_i32", "Eltwise"},
      {"opencl/generated/eltwise_logical_unary_bool", "Eltwise"},
      {"opencl/generated/eltwise_logical_binary_bool", "Eltwise"},
      {"opencl/generated/eltwise_logical_binary_broadcast_bool", "Eltwise"},
      {"opencl/generated/eltwise_compare_f32", "Eltwise"},
      {"opencl/generated/eltwise_compare_broadcast_f32", "Eltwise"},
      {"opencl/generated/eltwise_select_f32", "Eltwise"},
      {"opencl/generated/eltwise_select_broadcast_f32", "Eltwise"},
      {"opencl/generated/eltwise_select_f16_dynamic", "Eltwise"},
      {"opencl/generated/pool2d_f32", "Pooling"},
      {"opencl/generated/pool2d_f16", "Pooling"},
      {"opencl/generated/interpolate_f32", "Interpolate"},
      {"opencl/generated/interpolate_f16", "Interpolate"},
      {"opencl/generated/reduction_f32", "Reduction"},
      {"opencl/generated/reduction_bool", "Reduction"},
      {"opencl/generated/softmax_f32", "Softmax"},
      {"opencl/generated/softmax_f16", "Softmax"},
      {"opencl/generated/softmax_f32_dynamic_static_rank", "Softmax"},
      {"opencl/generated/softmax_f16_dynamic_static_rank", "Softmax"},
  };
  return specs;
}

const std::vector<OpenClOperationSupportEntry> &
opencl_operation_support_entries() {
  static const std::vector<OpenClOperationSupportEntry> entries = {
      {is_opencl_range_node, query_opencl_range_operation, nullptr},
      {is_opencl_tile_node, query_opencl_tile_operation, nullptr},
      {is_opencl_softmax_node, query_opencl_softmax_operation, nullptr},
      {is_opencl_conv2d_node, query_opencl_conv2d_operation, nullptr},
      {is_opencl_interpolate_node, query_opencl_interpolate_operation, nullptr},
      {is_matmul_node, nullptr, "missing_opencl_matmul_kernel_unit"},
      {is_opencl_activation_node, query_opencl_activation_operation, nullptr},
      {is_opencl_eltwise_node, query_opencl_eltwise_operation, nullptr},
      {is_opencl_reduction_node, query_opencl_reduction_operation, nullptr},
      {is_opencl_pool2d_node, query_opencl_pool2d_operation, nullptr},
      {is_transpose_node, nullptr, "missing_opencl_transpose_kernel_unit"},
      {is_opencl_shapeof_node, query_opencl_shapeof_operation, nullptr},
      {is_concat_node, nullptr, "missing_opencl_concat_kernel_unit"},
      {is_split_node, nullptr, "missing_opencl_split_kernel_unit"},
  };
  return entries;
}

const std::vector<OpenClArtifactFamilyEntry> &opencl_artifact_family_entries() {
  static const std::vector<OpenClArtifactFamilyEntry> entries = {
      {is_opencl_conv2d_node, make_opencl_conv2d_source_artifact,
       build_opencl_conv2d_kernel_artifact_payload},
      {is_opencl_range_node, make_opencl_range_source_artifact,
       build_opencl_range_kernel_artifact_payload},
      {is_opencl_tile_node, make_opencl_tile_source_artifact,
       build_opencl_tile_kernel_artifact_payload},
      {is_opencl_softmax_node, make_opencl_softmax_source_artifact,
       build_opencl_softmax_kernel_artifact_payload},
      {is_opencl_pool2d_node, make_opencl_pool2d_source_artifact,
       build_opencl_pool2d_kernel_artifact_payload},
      {is_opencl_interpolate_node, make_opencl_interpolate_source_artifact,
       build_opencl_interpolate_kernel_artifact_payload},
      {is_opencl_shapeof_node, make_opencl_shapeof_source_artifact,
       build_opencl_shapeof_kernel_artifact_payload},
      {is_opencl_activation_node, make_opencl_activation_source_artifact,
       build_opencl_activation_kernel_artifact_payload},
      {is_opencl_eltwise_node, make_opencl_eltwise_source_artifact,
       build_opencl_eltwise_kernel_artifact_payload},
      {is_opencl_reduction_node, make_opencl_reduction_source_artifact,
       build_opencl_reduction_kernel_artifact_payload},
  };
  return entries;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
