// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mlir/msl_codegen_apple_msl_shape.hpp"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "mlir/codegen_common.hpp"
#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/msl_codegen_apple_msl_binding.hpp"
#include "mlir/msl_codegen_apple_msl_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reverse.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/tile.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

std::vector<int32_t>
make_shapeof_msl_scalar_args(const std::shared_ptr<const ov::Node> &node) {
  OPENVINO_ASSERT(node, "GFX Metal ShapeOf: node is null");
  const auto input_pshape = node->get_input_partial_shape(0);
  OPENVINO_ASSERT(input_pshape.rank().is_static(),
                  "GFX Metal ShapeOf: input rank must be static");
  return {static_cast<int32_t>(input_pshape.rank().get_length())};
}

KernelSource make_shapeof_msl_kernel_source(
    const std::shared_ptr<const ov::Node> &node, mlir::ModuleOp module) {
  OPENVINO_ASSERT(node, "GFX Metal ShapeOf: node is null");
  ShapeOfCodegenDesc desc{};
  const auto input_pshape = node->get_input_partial_shape(0);
  OPENVINO_ASSERT(input_pshape.rank().is_static(),
                  "GFX Metal ShapeOf: input rank must be static");
  desc.rank = static_cast<uint32_t>(input_pshape.rank().get_length());
  desc.element_type = node->get_output_element_type(0);
  return make_kernel_source(module, "shapeof_kernel",
                            generate_msl_from_mlir(module, desc));
}

std::vector<int32_t>
make_tile_msl_scalar_args(const std::shared_ptr<const ov::Node> &node) {
  OPENVINO_ASSERT(node, "GFX Metal Tile: node is null");
  const auto output_pshape = node->get_output_partial_shape(0);
  const auto rank = output_pshape.rank().is_static()
                        ? static_cast<int32_t>(std::max<int64_t>(
                              output_pshape.rank().get_length(), 1))
                        : 1;
  const auto output_count =
      output_pshape.is_static()
          ? static_cast<int32_t>(ov::shape_size(output_pshape.to_shape()))
          : 0;
  return {output_count, rank};
}

KernelSource make_tile_msl_kernel_source(
    const std::shared_ptr<const ov::Node> &node, mlir::ModuleOp module) {
  OPENVINO_ASSERT(node, "GFX Metal Tile: node is null");
  TileCodegenDesc desc{};
  desc.element_type = node->get_output_element_type(0);
  auto source = make_kernel_source(module, "tile_kernel",
                                   generate_msl_from_mlir(module, desc));
  return source;
}

std::vector<int32_t>
make_range_msl_scalar_args(const std::shared_ptr<const ov::Node> &node) {
  OPENVINO_ASSERT(node, "GFX Metal Range: node is null");
  const auto output_count =
      node->get_output_partial_shape(0).is_static()
          ? static_cast<int32_t>(ov::shape_size(node->get_output_shape(0)))
          : int32_t{0};
  return {output_count};
}

KernelSource make_range_msl_kernel_source(
    const std::shared_ptr<const ov::Node> &node, mlir::ModuleOp module) {
  OPENVINO_ASSERT(node, "GFX Metal Range: node is null");
  RangeCodegenDesc desc{};
  desc.element_type = node->get_output_element_type(0);
  desc.output_type = node->get_output_element_type(0);
  desc.start_type = node->get_input_element_type(0);
  desc.stop_type = node->get_input_element_type(1);
  desc.step_type = node->get_input_element_type(2);
  auto source = make_kernel_source(module, "range_kernel",
                                   generate_msl_from_mlir(module, desc));
  return source;
}

} // namespace

GfxMslGeneratedKernelSourcePlan make_shapeof_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node, mlir::ModuleOp module) {
  auto binding = make_backend_custom_kernel_binding_plan(
      "ShapeOf", "shapeof_kernel", make_shapeof_msl_scalar_args(node));
  if (!binding.valid) {
    return {};
  }

  auto plan = make_msl_generated_custom_kernel_source_plan(
      make_shapeof_msl_kernel_source(node, module), binding);
  // The source plan owns the exact ABI; keep stale typed-program attrs out of
  // the direct compile path after the concrete MSL has been generated.
  plan.source.module = {};
  return plan;
}

GfxMslGeneratedKernelSourcePlan make_tile_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node, mlir::ModuleOp module) {
  auto binding = make_backend_custom_kernel_binding_plan(
      "Tile", "tile_kernel", make_tile_msl_scalar_args(node));
  if (!binding.valid) {
    return {};
  }

  auto plan = make_msl_generated_custom_kernel_source_plan(
      make_tile_msl_kernel_source(node, module), binding);
  // The source plan owns the exact ABI; keep stale typed-program attrs out of
  // the direct compile path after the concrete MSL has been generated.
  plan.source.module = {};
  return plan;
}

GfxMslGeneratedKernelSourcePlan make_range_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node, mlir::ModuleOp module) {
  auto binding = make_backend_custom_kernel_binding_plan(
      "Range", "range_kernel", make_range_msl_scalar_args(node));
  if (!binding.valid) {
    return {};
  }

  auto plan = make_msl_generated_custom_kernel_source_plan(
      make_range_msl_kernel_source(node, module), binding);
  // The source plan owns the exact ABI; keep stale typed-program attrs out of
  // the direct compile path after the concrete MSL has been generated.
  plan.source.module = {};
  return plan;
}

std::optional<KernelSource> make_apple_metal_shape_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return std::nullopt;
  }

  if (std::dynamic_pointer_cast<const ov::op::v0::ShapeOf>(node) ||
      std::dynamic_pointer_cast<const ov::op::v3::ShapeOf>(node)) {
    source = make_shapeof_msl_kernel_source(node, source.module);
    require_apple_msl_generated_kernel_source_binding(
        source, "ShapeOf", "shapeof_kernel", make_shapeof_msl_scalar_args(node));
    return source;
  }

  if (auto pad = std::dynamic_pointer_cast<const ov::op::v1::Pad>(node)) {
    PadCodegenDesc desc{};
    desc.element_type = pad->get_output_element_type(0);
    if (auto c = ov::as_type_ptr<const ov::op::v0::Constant>(
            pad->input_value(3).get_node_shared_ptr())) {
      if (c->get_element_type().is_real()) {
        desc.pad_value = c->cast_vector<double>()[0];
      } else if (c->get_element_type().is_integral_number()) {
        desc.pad_value = c->cast_vector<int64_t>()[0];
      }
    }
    source.entry_point = "pad_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    return source;
  }

  if (std::dynamic_pointer_cast<const ov::op::v0::Tile>(node)) {
    source = make_tile_msl_kernel_source(node, source.module);
    require_apple_msl_generated_kernel_source_binding(
        source, "Tile", "tile_kernel", make_tile_msl_scalar_args(node));
    return source;
  }

  if (std::dynamic_pointer_cast<const ov::op::v1::Broadcast>(node) ||
      std::dynamic_pointer_cast<const ov::op::v3::Broadcast>(node)) {
    BroadcastCodegenDesc desc{};
    desc.element_type = node->get_output_element_type(0);
    desc.has_target_shape_input =
        source.module && get_entry_func(source.module) &&
        get_entry_func(source.module).getNumArguments() > 1;
    source.entry_point = "broadcast_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    const auto input_shape = shape_from_entry_argument_or_partial(
        source.module, 0, node->get_input_partial_shape(0));
    const auto output_shape = output_shape_for_codegen(source.module, node);
    require_apple_msl_generated_kernel_source_binding(
        source, "Broadcast", "broadcast_kernel",
        {static_cast<int32_t>(ov::shape_size(output_shape)),
         static_cast<int32_t>(output_shape.size()),
         static_cast<int32_t>(input_shape.size())});
    if (desc.has_target_shape_input) {
      auto plan = make_backend_custom_kernel_roles_binding_plan(
          "Broadcast", "broadcast_kernel",
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
           GfxKernelBufferRole::TensorOutput,
           GfxKernelBufferRole::ScalarParam,
           GfxKernelBufferRole::ScalarParam,
           GfxKernelBufferRole::ScalarParam,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams});
      OPENVINO_ASSERT(configure_backend_custom_kernel_source_from_binding_plan(
                          source, plan),
                      "GFX Metal Broadcast: failed to configure dynamic "
                      "target-shape binding");
    }
    return source;
  }

  if (std::dynamic_pointer_cast<const ov::op::v4::Range>(node)) {
    source = make_range_msl_kernel_source(node, source.module);
    require_apple_msl_generated_kernel_source_binding(
        source, "Range", "range_kernel", make_range_msl_scalar_args(node));
    return source;
  }

  if (auto reverse =
          std::dynamic_pointer_cast<const ov::op::v1::Reverse>(node)) {
    ReverseCodegenDesc desc{};
    const auto in = reverse->get_input_shape(0);
    desc.element_type = reverse->get_output_element_type(0);
    desc.rank = static_cast<uint32_t>(in.size());
    desc.total = static_cast<uint32_t>(ov::shape_size(in));
    uint32_t stride = 1;
    for (int i = static_cast<int>(in.size()) - 1; i >= 0; --i) {
      desc.strides[static_cast<size_t>(i)] = stride;
      desc.dims[static_cast<size_t>(i)] =
          static_cast<uint32_t>(in[static_cast<size_t>(i)]);
      stride *= static_cast<uint32_t>(in[static_cast<size_t>(i)]);
    }
    auto axes_const = ov::as_type_ptr<const ov::op::v0::Constant>(
        reverse->input_value(1).get_node_shared_ptr());
    OPENVINO_ASSERT(axes_const, "Reverse axes must be constant");
    for (auto axis_value : axes_const->cast_vector<int64_t>()) {
      uint32_t axis = static_cast<uint32_t>(
          axis_value < 0 ? axis_value + in.size() : axis_value);
      desc.axes_mask |= (1u << axis);
    }
    source.entry_point = "reverse_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    return source;
  }

  return std::nullopt;
}

} // namespace gfx_plugin
} // namespace ov
