// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_common.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>

#include "mlir/codegen_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"

namespace ov {
namespace gfx_plugin {
ov::Shape static_shape_or_placeholder(const ov::PartialShape &pshape) {
  OPENVINO_ASSERT(pshape.rank().is_static(),
                  "GFX Metal: tensor rank must be static for MSL codegen");
  ov::Shape shape;
  shape.reserve(static_cast<size_t>(pshape.rank().get_length()));
  for (const auto &dim : pshape) {
    shape.push_back(dim.is_static() ? static_cast<size_t>(dim.get_length())
                                    : 1);
  }
  return shape;
}

std::vector<int64_t> to_i64_shape(const ov::Shape &shape) {
  std::vector<int64_t> values;
  values.reserve(shape.size());
  for (auto dim : shape) {
    values.push_back(static_cast<int64_t>(dim));
  }
  return values;
}

std::vector<int64_t> make_strides(const ov::Shape &shape) {
  const size_t rank = shape.size();
  std::vector<int64_t> strides(rank, 1);
  for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] *
        static_cast<int64_t>(shape[static_cast<size_t>(i + 1)]);
  }
  return strides;
}

void fill_broadcast_strides(const ov::Shape &output_shape,
                            const ov::Shape &input_shape,
                            std::vector<int64_t> &strides) {
  const size_t output_rank = output_shape.size();
  const size_t input_rank = input_shape.size();
  strides.assign(output_rank, 0);
  auto input_strides = make_strides(input_shape);
  for (size_t i = 0; i < output_rank; ++i) {
    const size_t output_dim = output_shape[output_rank - 1 - i];
    const size_t input_dim =
        (i < input_rank) ? input_shape[input_rank - 1 - i] : 1;
    const size_t input_stride =
        (i < input_rank) ? input_strides[input_rank - 1 - i] : 0;
    if (input_dim == output_dim) {
      strides[output_rank - 1 - i] = static_cast<int64_t>(input_stride);
    } else if (input_dim == 1) {
      strides[output_rank - 1 - i] = 0;
    } else {
      OPENVINO_THROW("GFX Metal: incompatible broadcast dims");
    }
  }
}

ov::Shape shape_from_entry_argument(mlir::ModuleOp module, size_t arg_idx,
                                    const ov::Shape &fallback) {
  if (!module) {
    return fallback;
  }
  auto func = get_entry_func(module);
  if (!func || arg_idx >= func.getNumArguments()) {
    return fallback;
  }
  auto type = func.getArgument(arg_idx).getType();
  if (auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
    if (!ranked.hasStaticShape()) {
      return fallback;
    }
    ov::Shape shape;
    shape.reserve(ranked.getRank());
    for (int64_t dim : ranked.getShape()) {
      shape.push_back(static_cast<size_t>(dim));
    }
    return shape;
  }
  if (auto memref = llvm::dyn_cast<mlir::MemRefType>(type)) {
    if (!memref.hasStaticShape()) {
      return fallback;
    }
    ov::Shape shape;
    shape.reserve(memref.getRank());
    for (int64_t dim : memref.getShape()) {
      shape.push_back(static_cast<size_t>(dim));
    }
    return shape;
  }
  return fallback;
}

ov::Shape
shape_from_entry_argument_or_partial(mlir::ModuleOp module, size_t arg_idx,
                                     const ov::PartialShape &fallback) {
  return shape_from_entry_argument(module, arg_idx,
                                   static_shape_or_placeholder(fallback));
}

ov::Shape
output_shape_for_codegen(mlir::ModuleOp module,
                         const std::shared_ptr<const ov::Node> &node) {
  OPENVINO_ASSERT(node, "GFX Metal: output shape requested for null node");
  if (node->get_output_partial_shape(0).is_static()) {
    return node->get_output_shape(0);
  }
  if (module) {
    auto func = get_entry_func(module);
    if (func && func.getFunctionType().getNumResults() > 0) {
      auto type = func.getFunctionType().getResult(0);
      if (auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
        ov::Shape shape;
        shape.reserve(ranked.getRank());
        for (int64_t dim : ranked.getShape()) {
          shape.push_back(
              dim == mlir::ShapedType::kDynamic ? 1 : static_cast<size_t>(dim));
        }
        return shape;
      }
      if (auto memref = llvm::dyn_cast<mlir::MemRefType>(type)) {
        ov::Shape shape;
        shape.reserve(memref.getRank());
        for (int64_t dim : memref.getShape()) {
          shape.push_back(
              dim == mlir::ShapedType::kDynamic ? 1 : static_cast<size_t>(dim));
        }
        return shape;
      }
    }
  }
  return static_shape_or_placeholder(node->get_output_partial_shape(0));
}

std::vector<int64_t> read_absorbed_input_permutation(mlir::ModuleOp module,
                                                     size_t input_idx) {
  std::vector<int64_t> permutation;
  if (!module) {
    return permutation;
  }
  auto attr = module->getAttrOfType<mlir::ArrayAttr>(
      "gfx.absorbed_input" + std::to_string(input_idx) + "_perm");
  if (!attr) {
    return permutation;
  }
  permutation.reserve(attr.size());
  for (auto value : attr) {
    auto int_attr = llvm::dyn_cast<mlir::IntegerAttr>(value);
    OPENVINO_ASSERT(
        int_attr, "GFX Metal: absorbed input permutation attr must be integer");
    permutation.push_back(int_attr.getInt());
  }
  return permutation;
}

std::optional<EltwiseKind> eltwise_kind_from_node(const ov::Node &node) {
  const std::string type = node.get_type_name();
  if (type == "Add")
    return EltwiseKind::Add;
  if (type == "Subtract")
    return EltwiseKind::Sub;
  if (type == "Multiply")
    return EltwiseKind::Mul;
  if (type == "Divide")
    return EltwiseKind::Div;
  if (type == "Power")
    return EltwiseKind::Pow;
  if (type == "Mod")
    return EltwiseKind::Mod;
  if (type == "FloorMod")
    return EltwiseKind::FloorMod;
  if (type == "PRelu")
    return EltwiseKind::Prelu;
  if (type == "SquaredDifference")
    return EltwiseKind::SquaredDiff;
  if (type == "Minimum")
    return EltwiseKind::Min;
  if (type == "Maximum")
    return EltwiseKind::Max;
  if (type == "LogicalAnd")
    return EltwiseKind::LogicalAnd;
  if (type == "LogicalOr")
    return EltwiseKind::LogicalOr;
  if (type == "LogicalXor")
    return EltwiseKind::LogicalXor;
  if (type == "Equal")
    return EltwiseKind::Equal;
  if (type == "NotEqual")
    return EltwiseKind::NotEqual;
  if (type == "Less")
    return EltwiseKind::Less;
  if (type == "Greater")
    return EltwiseKind::Greater;
  if (type == "LessEqual")
    return EltwiseKind::LessEqual;
  if (type == "GreaterEqual")
    return EltwiseKind::GreaterEqual;
  return std::nullopt;
}

std::optional<ReduceKind> reduce_kind_from_node(const ov::Node &node) {
  const std::string type = node.get_type_name();
  if (type == "ReduceSum")
    return ReduceKind::Sum;
  if (type == "ReduceMean")
    return ReduceKind::Mean;
  if (type == "ReduceMax")
    return ReduceKind::Max;
  if (type == "ReduceMin")
    return ReduceKind::Min;
  if (type == "ReduceProd")
    return ReduceKind::Prod;
  if (type == "ReduceL1")
    return ReduceKind::L1;
  if (type == "ReduceL2")
    return ReduceKind::L2;
  return std::nullopt;
}

std::optional<ActivationKind>
unary_activation_kind_from_node(const ov::Node &node) {
  const std::string type = node.get_type_name();
  if (type == "Relu")
    return ActivationKind::Relu;
  if (type == "Sigmoid")
    return ActivationKind::Sigmoid;
  if (type == "Tanh")
    return ActivationKind::Tanh;
  if (type == "Elu")
    return ActivationKind::Elu;
  if (type == "Gelu")
    return ActivationKind::Gelu;
  if (type == "Swish")
    return ActivationKind::Swish;
  if (type == "HSwish")
    return ActivationKind::HSwish;
  if (type == "HSigmoid")
    return ActivationKind::HSigmoid;
  if (type == "SoftPlus")
    return ActivationKind::SoftPlus;
  if (type == "Mish")
    return ActivationKind::Mish;
  if (type == "SoftSign")
    return ActivationKind::SoftSign;
  if (type == "Abs")
    return ActivationKind::Abs;
  if (type == "Sign")
    return ActivationKind::Sign;
  if (type == "Clamp")
    return ActivationKind::Clamp;
  if (type == "Exp")
    return ActivationKind::Exp;
  if (type == "Log")
    return ActivationKind::Log;
  if (type == "Sqrt")
    return ActivationKind::Sqrt;
  if (type == "Floor")
    return ActivationKind::Floor;
  if (type == "Ceiling" || type == "Ceil")
    return ActivationKind::Ceil;
  if (type == "Negative")
    return ActivationKind::Negative;
  if (type == "Sin")
    return ActivationKind::Sin;
  if (type == "Cos")
    return ActivationKind::Cos;
  if (type == "Tan")
    return ActivationKind::Tan;
  if (type == "Erf")
    return ActivationKind::Erf;
  if (type == "Asin")
    return ActivationKind::Asin;
  if (type == "Acos")
    return ActivationKind::Acos;
  if (type == "Atan")
    return ActivationKind::Atan;
  if (type == "Asinh")
    return ActivationKind::Asinh;
  if (type == "Acosh")
    return ActivationKind::Acosh;
  if (type == "Atanh")
    return ActivationKind::Atanh;
  if (type == "Sinh")
    return ActivationKind::Sinh;
  if (type == "Cosh")
    return ActivationKind::Cosh;
  if (type == "Round")
    return ActivationKind::RoundAway;
  return std::nullopt;
}

std::optional<ActivationKind>
activation_kind_from_module_attr(mlir::ModuleOp module,
                                 llvm::StringRef attr_name) {
  if (!module) {
    return std::nullopt;
  }
  auto attr = module->getAttrOfType<mlir::StringAttr>(attr_name);
  if (!attr) {
    return std::nullopt;
  }
  const auto value = attr.getValue();
  if (value == "Relu")
    return ActivationKind::Relu;
  if (value == "Sigmoid")
    return ActivationKind::Sigmoid;
  if (value == "Tanh")
    return ActivationKind::Tanh;
  if (value == "Gelu")
    return ActivationKind::Gelu;
  if (value == "Swish")
    return ActivationKind::Swish;
  if (value == "HSwish")
    return ActivationKind::HSwish;
  if (value == "HSigmoid")
    return ActivationKind::HSigmoid;
  return std::nullopt;
}

namespace {

std::vector<int64_t> get_slice_const_i64(const ov::Output<ov::Node> &source,
                                         const char *what) {
  auto c = ov::util::get_constant_from_source(source);
  OPENVINO_ASSERT(c, "GFX Metal Slice: ", what, " must be Constant");
  return c->cast_vector<int64_t>();
}

int64_t normalize_slice_index(int64_t index, int64_t dim, bool is_begin) {
  if (index < 0) {
    index += dim;
  }
  if (is_begin) {
    return std::clamp<int64_t>(index, 0, dim);
  }
  return std::clamp<int64_t>(index, -1, dim);
}

struct StaticSliceMeta {
  std::vector<uint32_t> out_shape;
  std::vector<uint32_t> in_stride;
  std::vector<int32_t> starts;
  std::vector<int32_t> steps;
  uint32_t total = 0;
};

StaticSliceMeta
build_static_slice_meta(const std::shared_ptr<const ov::Node> &node) {
  OPENVINO_ASSERT(node, "GFX Metal Slice: node is null");
  const auto in_shape = node->get_input_shape(0);
  const auto out_shape = node->get_output_shape(0);
  const size_t rank = in_shape.size();
  OPENVINO_ASSERT(
      rank == out_shape.size(),
      "GFX Metal Slice: rank-changing Slice/StridedSlice is not supported");

  StaticSliceMeta meta;
  meta.out_shape.reserve(rank);
  meta.starts.assign(rank, 0);
  meta.steps.assign(rank, 1);
  meta.in_stride.assign(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    meta.out_shape.push_back(static_cast<uint32_t>(out_shape[i]));
  }
  for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
    meta.in_stride[static_cast<size_t>(i)] =
        meta.in_stride[static_cast<size_t>(i + 1)] *
        static_cast<uint32_t>(in_shape[static_cast<size_t>(i + 1)]);
  }
  meta.total = static_cast<uint32_t>(ov::shape_size(out_shape));

  if (auto slice = ov::as_type_ptr<const ov::op::v8::Slice>(node)) {
    auto starts = get_slice_const_i64(slice->input_value(1), "Slice starts");
    auto ends = get_slice_const_i64(slice->input_value(2), "Slice ends");
    auto steps = get_slice_const_i64(slice->input_value(3), "Slice steps");
    std::vector<int64_t> axes;
    if (slice->get_input_size() > 4) {
      axes = get_slice_const_i64(slice->input_value(4), "Slice axes");
    } else {
      axes.resize(starts.size());
      std::iota(axes.begin(), axes.end(), 0);
    }
    OPENVINO_ASSERT(starts.size() == ends.size() &&
                        starts.size() == steps.size() &&
                        starts.size() == axes.size(),
                    "GFX Metal Slice: starts/ends/steps/axes size mismatch");
    for (size_t i = 0; i < axes.size(); ++i) {
      int64_t axis = axes[i];
      if (axis < 0) {
        axis += static_cast<int64_t>(rank);
      }
      OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank,
                      "GFX Metal Slice: axis out of range");
      OPENVINO_ASSERT(steps[i] != 0,
                      "GFX Metal Slice: zero step is not supported");
      const auto dim =
          static_cast<int64_t>(in_shape[static_cast<size_t>(axis)]);
      meta.starts[static_cast<size_t>(axis)] =
          static_cast<int32_t>(normalize_slice_index(starts[i], dim, true));
      meta.steps[static_cast<size_t>(axis)] = static_cast<int32_t>(steps[i]);
    }
    return meta;
  }

  auto slice = ov::as_type_ptr<const ov::op::v1::StridedSlice>(node);
  OPENVINO_ASSERT(slice, "GFX Metal Slice: expected Slice/StridedSlice node");
  OPENVINO_ASSERT(
      std::all_of(slice->get_new_axis_mask().begin(),
                  slice->get_new_axis_mask().end(),
                  [](int64_t v) { return v == 0; }),
      "GFX Metal Slice: StridedSlice new_axis_mask is not supported");
  OPENVINO_ASSERT(
      std::all_of(slice->get_shrink_axis_mask().begin(),
                  slice->get_shrink_axis_mask().end(),
                  [](int64_t v) { return v == 0; }),
      "GFX Metal Slice: StridedSlice shrink_axis_mask is not supported");
  OPENVINO_ASSERT(
      std::all_of(slice->get_ellipsis_mask().begin(),
                  slice->get_ellipsis_mask().end(),
                  [](int64_t v) { return v == 0; }),
      "GFX Metal Slice: StridedSlice ellipsis_mask is not supported");

  auto begin = get_slice_const_i64(slice->input_value(1), "StridedSlice begin");
  auto end = get_slice_const_i64(slice->input_value(2), "StridedSlice end");
  std::vector<int64_t> strides(rank, 1);
  if (slice->get_input_size() > 3) {
    auto values =
        get_slice_const_i64(slice->input_value(3), "StridedSlice strides");
    OPENVINO_ASSERT(values.size() <= rank,
                    "GFX Metal Slice: StridedSlice strides rank mismatch");
    std::copy(values.begin(), values.end(), strides.begin());
  }
  const auto &begin_mask = slice->get_begin_mask();
  const auto &end_mask = slice->get_end_mask();
  for (size_t axis = 0; axis < rank; ++axis) {
    const auto dim = static_cast<int64_t>(in_shape[axis]);
    const bool masked_begin = axis < begin_mask.size() && begin_mask[axis] != 0;
    const bool masked_end = axis < end_mask.size() && end_mask[axis] != 0;
    const int64_t step = strides[axis];
    OPENVINO_ASSERT(step != 0,
                    "GFX Metal Slice: StridedSlice zero step is not supported");
    int64_t start = axis < begin.size() ? begin[axis] : 0;
    int64_t finish = axis < end.size() ? end[axis] : dim;
    start = masked_begin ? (step < 0 ? dim - 1 : 0)
                         : normalize_slice_index(start, dim, true);
    finish = masked_end ? (step < 0 ? -1 : dim)
                        : normalize_slice_index(finish, dim, false);
    (void)finish;
    meta.starts[axis] = static_cast<int32_t>(start);
    meta.steps[axis] = static_cast<int32_t>(step);
  }
  return meta;
}

} // namespace

std::string
generate_static_msl_for_slice(const std::shared_ptr<const ov::Node> &node,
                              const ov::element::Type &storage_type) {
  const auto meta = build_static_slice_meta(node);
  const auto scalar_t = msl_type_from_element(
      storage_type == ov::element::dynamic ? ov::element::f32 : storage_type);
  const uint32_t rank = static_cast<uint32_t>(meta.out_shape.size());
  std::ostringstream ss;
  ss << "#include <metal_stdlib>\nusing namespace metal;\n";
  ss << "using scalar_t = " << scalar_t << ";\n";
  ss << "constant uint TOTAL_C = " << meta.total << ";\n";
  ss << "constant uint RANK_C = " << rank << ";\n";
  ss << "constant uint OUT_SHAPE_C[" << rank << "] = {";
  for (size_t i = 0; i < meta.out_shape.size(); ++i) {
    if (i)
      ss << ", ";
    ss << meta.out_shape[i];
  }
  ss << "};\n";
  ss << "constant uint IN_STRIDE_C[" << rank << "] = {";
  for (size_t i = 0; i < meta.in_stride.size(); ++i) {
    if (i)
      ss << ", ";
    ss << meta.in_stride[i];
  }
  ss << "};\n";
  ss << "constant int STARTS_C[" << rank << "] = {";
  for (size_t i = 0; i < meta.starts.size(); ++i) {
    if (i)
      ss << ", ";
    ss << meta.starts[i];
  }
  ss << "};\n";
  ss << "constant int STEPS_C[" << rank << "] = {";
  for (size_t i = 0; i < meta.steps.size(); ++i) {
    if (i)
      ss << ", ";
    ss << meta.steps[i];
  }
  ss << "};\n";
  ss << "kernel void slice_kernel(\n";
  ss << "  device const scalar_t* A [[buffer(0)]],\n";
  ss << "  device scalar_t* C [[buffer(1)]],\n";
  ss << "  uint gid [[thread_position_in_grid]]) {\n";
  ss << "    if (gid >= TOTAL_C) return;\n";
  ss << "    uint idx = gid;\n";
  ss << "    int in_off = 0;\n";
  ss << "    for (int d = (int)RANK_C - 1; d >= 0; --d) {\n";
  ss << "        uint coord = idx % OUT_SHAPE_C[d];\n";
  ss << "        idx /= OUT_SHAPE_C[d];\n";
  ss << "        in_off += (STARTS_C[d] + int(coord) * STEPS_C[d]) * "
        "int(IN_STRIDE_C[d]);\n";
  ss << "    }\n";
  ss << "    C[gid] = A[in_off];\n";
  ss << "}\n";
  return ss.str();
}
} // namespace gfx_plugin
} // namespace ov
