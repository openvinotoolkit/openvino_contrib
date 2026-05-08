// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_ops.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "mlir/msl_codegen.hpp"
#include "mlir/msl_codegen_apple_msl.hpp"
#include "mlir/msl_codegen_apple_msl_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reverse.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/space_to_depth.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {

bool configure_apple_metal_slice_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node,
    const ov::element::Type &storage_type, bool has_runtime_slice_params) {
  if (!node || (!ov::is_type<const ov::op::v8::Slice>(node) &&
                !ov::is_type<const ov::op::v1::StridedSlice>(node))) {
    return false;
  }

  const ov::element::Type effective_type =
      storage_type == ov::element::dynamic ? node->get_output_element_type(0)
                                           : storage_type;
  ConvertCodegenDesc desc{};
  desc.element_type = effective_type;
  desc.dst_type = effective_type;
  source.entry_point = "slice_kernel";
  const bool dynamic_slice_shape =
      !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static();
  if (!has_runtime_slice_params && !dynamic_slice_shape) {
    source.msl_source = generate_static_msl_for_slice(node, desc.dst_type);
    source.msl_generator = {};
    source.module = {};
    return true;
  }

  source.msl_source.clear();
  source.msl_generator = [desc](mlir::ModuleOp module) mutable {
    return generate_msl_for_slice_generic(desc, module);
  };
  return true;
}

bool configure_apple_metal_elementwise_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return false;
  }

  const std::string type = node->get_type_name();
  if (type == "Select") {
    const auto out_shape = output_shape_for_codegen(source.module, node);
    source.entry_point = "select_kernel";
    source.msl_generator = [element_type = node->get_output_element_type(0)](
                               mlir::ModuleOp module) {
      return generate_msl_for_select(module, element_type);
    };
    if (source.module) {
      const std::vector<int32_t> scalars{
          static_cast<int32_t>(ov::shape_size(out_shape)),
          static_cast<int32_t>(out_shape.empty() ? 1 : out_shape.size())};
      require_apple_msl_custom_kernel_binding(source.module, "Select",
                                              "select_kernel", scalars);
    }
    return true;
  }

  auto kind = eltwise_kind_from_node(*node);
  if (!kind) {
    return false;
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
  if (source.module) {
    std::vector<int32_t> scalars = {static_cast<int32_t>(desc.num_elements),
                                    static_cast<int32_t>(out_shape.size())};
    require_apple_msl_custom_kernel_binding(source.module, type,
                                            "eltwise_kernel", scalars);
  }
  return true;
}

bool configure_apple_metal_structural_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return false;
  }

  const auto reduce_kind = reduce_kind_from_node(*node);
  if (reduce_kind) {
    ReduceCodegenDesc desc{};
    desc.element_type = node->get_output_element_type(0);
    desc.kind = *reduce_kind;
    source.entry_point = "reduce_kernel";
    source.signature.output_arg_count = 1;
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    if (source.module) {
      require_apple_msl_custom_kernel_binding(
          source.module, node->get_type_name(), "reduce_kernel");
    }
    return true;
  }

  if (auto concat = std::dynamic_pointer_cast<const ov::op::v0::Concat>(node)) {
    ConcatCodegenDesc desc{};
    desc.element_type = concat->get_output_element_type(0);
    const auto out_pshape = concat->get_output_partial_shape(0);
    OPENVINO_ASSERT(out_pshape.rank().is_static(),
                    "GFX Metal Concat: output rank must be static");
    const size_t rank = static_cast<size_t>(out_pshape.rank().get_length());
    OPENVINO_ASSERT(rank > 0, "GFX Metal Concat: output rank must be positive");
    const size_t axis =
        normalize_axis(concat->get_axis(), rank, "GFX Metal Concat");
    (void)axis;
    desc.inner = 1;
    desc.outer = 1;
    desc.axis_total = 1;
    source.entry_point = "concat_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    return true;
  }

  if (auto split = std::dynamic_pointer_cast<const ov::op::v1::Split>(node)) {
    SplitCodegenDesc desc{};
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
    source.entry_point = "split_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    return true;
  }

  if (auto split =
          std::dynamic_pointer_cast<const ov::op::v1::VariadicSplit>(node)) {
    SplitCodegenDesc desc{};
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
    source.entry_point = "split_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    return true;
  }

  if (auto d2s =
          std::dynamic_pointer_cast<const ov::op::v0::DepthToSpace>(node)) {
    DepthToSpaceCodegenDesc desc{};
    const auto in = d2s->get_input_shape(0);
    const auto out = d2s->get_output_shape(0);
    desc.element_type = d2s->get_output_element_type(0);
    desc.N = in[0];
    desc.C = in[1];
    desc.H = in[2];
    desc.W = in[3];
    desc.block = static_cast<uint32_t>(d2s->get_block_size());
    desc.C_out = out[1];
    desc.H_out = out[2];
    desc.W_out = out[3];
    desc.mode = d2s->get_mode() ==
                        ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST
                    ? 0
                    : 1;
    desc.total = static_cast<uint32_t>(ov::shape_size(out));
    source.entry_point = "depth_to_space_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    return true;
  }

  if (auto s2d =
          std::dynamic_pointer_cast<const ov::op::v0::SpaceToDepth>(node)) {
    SpaceToDepthCodegenDesc desc{};
    const auto in = s2d->get_input_shape(0);
    const auto out = s2d->get_output_shape(0);
    desc.element_type = s2d->get_output_element_type(0);
    desc.N = in[0];
    desc.C = in[1];
    desc.H = in[2];
    desc.W = in[3];
    desc.block = static_cast<uint32_t>(s2d->get_block_size());
    desc.C_out = out[1];
    desc.H_out = out[2];
    desc.W_out = out[3];
    desc.mode = s2d->get_mode() ==
                        ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST
                    ? 0
                    : 1;
    desc.total = static_cast<uint32_t>(ov::shape_size(out));
    source.entry_point = "space_to_depth_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    return true;
  }

  if (auto transpose =
          std::dynamic_pointer_cast<const ov::op::v1::Transpose>(node)) {
    TransposeCodegenDesc desc{};
    desc.element_type = transpose->get_output_element_type(0);
    auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(
        transpose->input_value(1).get_node_shared_ptr());
    OPENVINO_ASSERT(perm_const, "Transpose perm must be constant");
    auto perm = perm_const->cast_vector<int64_t>();
    for (auto value : perm) {
      desc.perm.push_back(static_cast<uint32_t>(value));
    }
    source.entry_point = "transpose_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    if (source.module) {
      require_apple_msl_custom_kernel_binding(source.module, "Transpose",
                                              "transpose_kernel");
    }
    return true;
  }

  if (auto convert =
          std::dynamic_pointer_cast<const ov::op::v0::Convert>(node)) {
    ConvertCodegenDesc desc{};
    desc.src_type = convert->get_input_element_type(0);
    desc.dst_type = convert->get_output_element_type(0);
    desc.element_type = desc.dst_type == ov::element::dynamic ? ov::element::f32
                                                              : desc.dst_type;
    source.entry_point = "convert_kernel";
    source.signature.output_arg_count = 1;
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    if (source.module) {
      require_apple_msl_custom_kernel_binding(source.module, "Convert",
                                              "convert_kernel");
    }
    return true;
  }

  if (std::dynamic_pointer_cast<const ov::op::v0::ShapeOf>(node) ||
      std::dynamic_pointer_cast<const ov::op::v3::ShapeOf>(node)) {
    ShapeOfCodegenDesc desc{};
    const auto input_pshape = node->get_input_partial_shape(0);
    OPENVINO_ASSERT(input_pshape.rank().is_static(),
                    "ShapeOf: input rank must be static");
    desc.rank = static_cast<uint32_t>(input_pshape.rank().get_length());
    desc.element_type = node->get_output_element_type(0);
    source.entry_point = "shapeof_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    return true;
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
    return true;
  }

  if (std::dynamic_pointer_cast<const ov::op::v0::Tile>(node)) {
    TileCodegenDesc desc{};
    desc.element_type = node->get_output_element_type(0);
    source.entry_point = "tile_kernel";
    source.signature.output_arg_count = 1;
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    return true;
  }

  if (std::dynamic_pointer_cast<const ov::op::v1::Broadcast>(node) ||
      std::dynamic_pointer_cast<const ov::op::v3::Broadcast>(node)) {
    BroadcastCodegenDesc desc{};
    desc.element_type = node->get_output_element_type(0);
    source.entry_point = "broadcast_kernel";
    source.signature.output_arg_count = 1;
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    if (source.module) {
      require_apple_msl_custom_kernel_binding(source.module, "Broadcast",
                                              "broadcast_kernel");
    }
    return true;
  }

  if (std::dynamic_pointer_cast<const ov::op::v4::Range>(node)) {
    RangeCodegenDesc desc{};
    desc.element_type = node->get_output_element_type(0);
    desc.output_type = node->get_output_element_type(0);
    desc.start_type = node->get_input_element_type(0);
    desc.stop_type = node->get_input_element_type(1);
    desc.step_type = node->get_input_element_type(2);
    source.entry_point = "range_kernel";
    source.signature.output_arg_count = 1;
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    if (source.module) {
      require_apple_msl_custom_kernel_binding(source.module, "Range",
                                              "range_kernel");
    }
    return true;
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
    return true;
  }

  if (auto topk =
          std::dynamic_pointer_cast<const ov::op::util::TopKBase>(node)) {
    TopKCodegenDesc desc{};
    const auto in = topk->get_input_shape(0);
    const int64_t axis_i64 =
        normalize_axis(topk->get_axis(), in.size(), "TopK");
    const size_t axis = static_cast<size_t>(axis_i64);
    desc.axis_len = static_cast<uint32_t>(in[axis]);
    desc.k = static_cast<uint32_t>(topk->get_k());
    uint32_t outer = 1;
    uint32_t inner = 1;
    for (size_t i = 0; i < axis; ++i)
      outer *= static_cast<uint32_t>(in[i]);
    for (size_t i = axis + 1; i < in.size(); ++i)
      inner *= static_cast<uint32_t>(in[i]);
    desc.outer = outer;
    desc.inner = inner;
    desc.mode_max = topk->get_mode() == ov::op::TopKMode::MAX;
    switch (topk->get_sort_type()) {
    case ov::op::TopKSortType::SORT_INDICES:
      desc.sort_type = TopKSortType::SortIndices;
      break;
    case ov::op::TopKSortType::NONE:
      desc.sort_type = TopKSortType::None;
      break;
    case ov::op::TopKSortType::SORT_VALUES:
    default:
      desc.sort_type = TopKSortType::SortValues;
      break;
    }
    desc.element_type = topk->get_output_element_type(0);
    desc.index_type = topk->get_output_element_type(1);
    source.entry_point = "topk_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    return true;
  }

  return false;
}

} // namespace gfx_plugin
} // namespace ov
