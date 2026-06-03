// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/msl_codegen_apple_msl_binding.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_split.hpp"

#include <cstdint>
#include <memory>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "mlir/codegen_common.hpp"
#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/mlir_kernel_plan_utils.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

ConcatCodegenDesc make_concat_msl_codegen_desc(
    const std::shared_ptr<const ov::op::v0::Concat> &concat) {
  OPENVINO_ASSERT(concat, "GFX Metal Concat: node is null");
  ConcatCodegenDesc desc{};
  desc.element_type = concat->get_output_element_type(0);
  const auto out_pshape = concat->get_output_partial_shape(0);
  OPENVINO_ASSERT(out_pshape.rank().is_static(),
                  "GFX Metal Concat: output rank must be static");
  const size_t rank = static_cast<size_t>(out_pshape.rank().get_length());
  OPENVINO_ASSERT(rank > 0, "GFX Metal Concat: output rank must be positive");
  const size_t axis =
      normalize_axis(concat->get_axis(), rank, "GFX Metal Concat");
  desc.inner = 1;
  desc.outer = 1;
  desc.axis_total = 1;
  const auto out_shape = out_pshape.to_shape();
  for (size_t i = 0; i < axis; ++i) {
    desc.outer *= out_shape[i];
  }
  for (size_t i = axis + 1; i < out_shape.size(); ++i) {
    desc.inner *= out_shape[i];
  }
  desc.axis_total = out_shape[axis];
  desc.input_axis_lengths.reserve(concat->get_input_size());
  for (size_t input_idx = 0; input_idx < concat->get_input_size();
       ++input_idx) {
    OPENVINO_ASSERT(concat->get_input_partial_shape(input_idx).is_static(),
                    "GFX Metal Concat: input shape must be static");
    const auto input_shape = concat->get_input_shape(input_idx);
    OPENVINO_ASSERT(input_shape.size() == rank,
                    "GFX Metal Concat: input rank mismatch");
    desc.input_axis_lengths.push_back(input_shape[axis]);
  }
  return desc;
}

} // namespace

GfxMslGeneratedKernelSourcePlan make_direct_concat_msl_kernel_source_plan(
    KernelSource source, const ConcatCodegenDesc &desc) {
  if (desc.input_axis_lengths.empty() || desc.outer == 0 || desc.inner == 0 ||
      desc.axis_total == 0) {
    return {};
  }

  auto binding = make_backend_custom_kernel_direct_io_binding_plan(
      "Concat", "concat_kernel", desc.input_axis_lengths.size(), 1);
  if (!binding.valid) {
    return {};
  }

  source.entry_point = "concat_kernel";
  source.msl_source.clear();
  source.msl_generator = [desc](mlir::ModuleOp module) mutable {
    return generate_msl_from_mlir(module, desc);
  };
  return make_msl_generated_custom_kernel_source_plan(std::move(source),
                                                      binding);
}

GfxMslGeneratedKernelSourcePlan make_concat_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node, mlir::ModuleOp module) {
  auto concat = std::dynamic_pointer_cast<const ov::op::v0::Concat>(node);
  if (!concat) {
    return {};
  }
  const auto desc = make_concat_msl_codegen_desc(concat);
  auto binding = make_backend_custom_kernel_direct_io_binding_plan(
      "Concat", "concat_kernel", desc.input_axis_lengths.size(), 1);
  if (!binding.valid) {
    return {};
  }

  auto source = make_kernel_source(module, "concat_kernel",
                                   generate_msl_from_mlir(module, desc));
  auto plan =
      make_msl_generated_custom_kernel_source_plan(std::move(source), binding);
  // The direct source-plan owns the concrete MSL and ABI; do not leave stale
  // MLIR attrs on the compile path after source generation.
  plan.source.module = {};
  return plan;
}

GfxMslGeneratedKernelSourcePlan make_direct_split_msl_kernel_source_plan(
    std::string_view stage_type, const ov::element::Type &element_type,
    const ov::Shape &input_shape, const std::vector<size_t> &split_sizes,
    uint32_t axis_len, uint32_t inner_stride, mlir::ModuleOp module) {
  if (input_shape.empty() || split_sizes.empty() || axis_len == 0 ||
      inner_stride == 0) {
    return {};
  }

  const auto binding = make_backend_custom_kernel_direct_io_binding_plan(
      stage_type, "split_kernel", 1, split_sizes.size());
  if (!binding.valid) {
    return {};
  }
  mlir::ModuleOp manifest_module;
  if (module) {
    manifest_module =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(module.getContext()));
  }

  const auto total_elems = ov::shape_size(input_shape);
  const auto scalar = msl_type_from_element(element_type);
  std::ostringstream msl;
  msl << "#include <metal_stdlib>\nusing namespace metal;\n";
  msl << "constant uint OFFSETS[" << (split_sizes.size() + 1) << "] = {0";
  uint64_t prefix = 0;
  for (auto sz : split_sizes) {
    prefix += static_cast<uint64_t>(sz);
    msl << ", " << prefix;
  }
  msl << "};\n";
  msl << "constant uint AXIS_DIM = " << axis_len << ";\n";
  msl << "constant uint STRIDE_AFTER = " << inner_stride << ";\n";
  msl << "constant uint OUTER_STRIDE = AXIS_DIM * STRIDE_AFTER;\n";
  msl << "kernel void split_kernel(\n";
  msl << "  device const " << scalar << "* input [[buffer(0)]],\n";
  for (size_t oi = 0; oi < split_sizes.size(); ++oi) {
    msl << "  device " << scalar << "* out" << oi << " [[buffer(" << (oi + 1)
        << ")]],\n";
  }
  msl << "  uint gid [[thread_position_in_grid]]) {\n";
  msl << "    uint total = " << static_cast<uint32_t>(total_elems) << ";\n";
  msl << "    if (gid >= total) return;\n";
  msl << "    uint axis_idx = (gid / STRIDE_AFTER) % AXIS_DIM;\n";
  msl << "    uint outer = gid / OUTER_STRIDE;\n";
  msl << "    uint inner = gid % STRIDE_AFTER;\n";
  msl << "    uint o = 0;\n";
  msl << "    while (o + 1 < " << (split_sizes.size() + 1)
      << " && axis_idx >= OFFSETS[o + 1]) ++o;\n";
  msl << "    uint local_axis = axis_idx - OFFSETS[o];\n";
  msl << "    uint dst_axis_extent = OFFSETS[o + 1] - OFFSETS[o];\n";
  msl << "    uint dst_idx = (outer * dst_axis_extent + local_axis) * "
         "STRIDE_AFTER + inner;\n";
  msl << "    switch (o) {\n";
  for (size_t oi = 0; oi < split_sizes.size(); ++oi) {
    msl << "      case " << oi << ": out" << oi
        << "[dst_idx] = input[gid]; break;\n";
  }
  msl << "      default: break;\n";
  msl << "    }\n";
  msl << "}\n";

  auto source = make_kernel_source(manifest_module, "split_kernel", msl.str());
  return make_msl_generated_custom_kernel_source_plan(std::move(source),
                                                      binding);
}

GfxMslGeneratedKernelSourcePlan make_split_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node, mlir::ModuleOp module) {
  if (!node ||
      (!std::dynamic_pointer_cast<const ov::op::v1::Split>(node) &&
       !std::dynamic_pointer_cast<const ov::op::v1::VariadicSplit>(node))) {
    return {};
  }
  if (!node->get_input_partial_shape(0).is_static() ||
      node->get_output_size() == 0) {
    return {};
  }

  const auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(
      node->input_value(1).get_node_shared_ptr());
  if (!axis_const || axis_const->get_shape().size() > 0) {
    return {};
  }
  if (std::dynamic_pointer_cast<const ov::op::v1::VariadicSplit>(node) &&
      !ov::as_type_ptr<const ov::op::v0::Constant>(
          node->input_value(2).get_node_shared_ptr())) {
    return {};
  }

  const auto input_shape = node->get_input_shape(0);
  const size_t rank = input_shape.size();
  if (rank == 0) {
    return {};
  }
  const auto axis = normalize_axis(axis_const->cast_vector<int64_t>().at(0),
                                   rank,
                                   "GFX Metal Split");
  uint64_t inner = 1;
  for (size_t i = axis + 1; i < input_shape.size(); ++i) {
    inner *= input_shape[i];
  }

  std::vector<size_t> split_sizes;
  split_sizes.reserve(node->get_output_size());
  for (size_t i = 0; i < node->get_output_size(); ++i) {
    const auto output_pshape = node->get_output_partial_shape(i);
    if (!output_pshape.is_static()) {
      return {};
    }
    const auto output_shape = output_pshape.to_shape();
    if (output_shape.size() != rank) {
      return {};
    }
    split_sizes.push_back(output_shape[axis]);
  }

  const uint64_t axis_total = std::accumulate(split_sizes.begin(),
                                             split_sizes.end(),
                                             uint64_t{0});
  if (axis_total != input_shape[axis]) {
    return {};
  }
  return make_direct_split_msl_kernel_source_plan(
      node->get_type_name(), node->get_output_element_type(0), input_shape,
      split_sizes, static_cast<uint32_t>(input_shape[axis]),
      static_cast<uint32_t>(inner), module);
}

std::optional<KernelSource> make_apple_metal_concat_split_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return std::nullopt;
  }

  if (auto concat = std::dynamic_pointer_cast<const ov::op::v0::Concat>(node)) {
    const auto desc = make_concat_msl_codegen_desc(concat);
    auto source_plan = make_direct_concat_msl_kernel_source_plan(
        std::move(source), desc);
    OPENVINO_ASSERT(source_plan.valid(),
                    "GFX Metal Concat: direct IO source plan is invalid");
    return std::move(source_plan.source);
  }

  if (auto split = std::dynamic_pointer_cast<const ov::op::v1::Split>(node)) {
    SplitCodegenDesc desc{};
    desc.element_type = split->get_output_element_type(0);
    auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(
        split->input_value(1).get_node_shared_ptr());
    OPENVINO_ASSERT(axis_const, "Split axis must be constant");
    desc.axis = axis_const->cast_vector<int64_t>().at(0);
    const auto in_shape = split->get_input_shape(0);
    const auto out_shape = split->get_output_shape(0);
    desc.input_shape = to_i64_shape(in_shape);
    desc.source_input_shape =
        to_i64_shape(shape_from_entry_argument(source.module, 0, in_shape));
    desc.input_permutation = read_absorbed_input_permutation(source.module, 0);
    const size_t axis = static_cast<size_t>(
        desc.axis < 0 ? desc.axis + in_shape.size() : desc.axis);
    desc.split_sizes.assign(split->get_output_size(), out_shape[axis]);
    uint64_t inner = 1;
    uint64_t outer = 1;
    for (size_t i = axis + 1; i < in_shape.size(); ++i)
      inner *= in_shape[i];
    for (size_t i = 0; i < axis; ++i)
      outer *= in_shape[i];
    desc.inner = inner;
    desc.outer = outer;
    if (desc.input_permutation.empty()) {
      auto source_plan = make_direct_split_msl_kernel_source_plan(
          "Split", desc.element_type, in_shape, desc.split_sizes,
          static_cast<uint32_t>(in_shape[axis]), static_cast<uint32_t>(inner),
          source.module);
      OPENVINO_ASSERT(source_plan.valid(),
                      "GFX Metal Split: direct IO source plan is invalid");
      return std::move(source_plan.source);
    }
    source.entry_point = "split_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    require_apple_msl_generated_kernel_source_binding(source, "Split",
                                                      "split_kernel");
    return source;
  }

  if (auto split =
          std::dynamic_pointer_cast<const ov::op::v1::VariadicSplit>(node)) {
    SplitCodegenDesc desc{};
    desc.element_type = split->get_output_element_type(0);
    auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(
        split->input_value(1).get_node_shared_ptr());
    OPENVINO_ASSERT(axis_const, "VariadicSplit axis must be constant");
    desc.axis = axis_const->cast_vector<int64_t>().at(0);
    const auto in_shape = split->get_input_shape(0);
    desc.input_shape = to_i64_shape(in_shape);
    desc.source_input_shape =
        to_i64_shape(shape_from_entry_argument(source.module, 0, in_shape));
    desc.input_permutation = read_absorbed_input_permutation(source.module, 0);
    auto lengths_const = ov::as_type_ptr<const ov::op::v0::Constant>(
        split->input_value(2).get_node_shared_ptr());
    OPENVINO_ASSERT(lengths_const, "VariadicSplit lengths must be constant");
    auto lengths = lengths_const->cast_vector<int64_t>();
    desc.split_sizes.assign(lengths.begin(), lengths.end());
    const size_t axis = static_cast<size_t>(
        desc.axis < 0 ? desc.axis + in_shape.size() : desc.axis);
    uint64_t inner = 1;
    uint64_t outer = 1;
    for (size_t i = axis + 1; i < in_shape.size(); ++i)
      inner *= in_shape[i];
    for (size_t i = 0; i < axis; ++i)
      outer *= in_shape[i];
    desc.inner = inner;
    desc.outer = outer;
    if (desc.input_permutation.empty()) {
      auto source_plan = make_direct_split_msl_kernel_source_plan(
          "VariadicSplit", desc.element_type, in_shape, desc.split_sizes,
          static_cast<uint32_t>(in_shape[axis]), static_cast<uint32_t>(inner),
          source.module);
      OPENVINO_ASSERT(
          source_plan.valid(),
          "GFX Metal VariadicSplit: direct IO source plan is invalid");
      return std::move(source_plan.source);
    }
    source.entry_point = "split_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    require_apple_msl_generated_kernel_source_binding(source, "VariadicSplit",
                                                      "split_kernel");
    return source;
  }

  return std::nullopt;
}

} // namespace gfx_plugin
} // namespace ov
