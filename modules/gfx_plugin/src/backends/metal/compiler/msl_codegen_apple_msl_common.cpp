// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/msl_codegen_apple_msl_common.hpp"

#include "mlir/codegen_common.hpp"
#include "openvino/core/except.hpp"

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

std::string msl_stable_tanh_expr(const std::string &x) {
  return "tanh(clamp(" + x + ", -20.0f, 20.0f))";
}

std::string msl_stable_gelu_tanh_expr(const std::string &x) {
  const std::string tanh_arg =
      "0.79788456f * (" + x + " + 0.044715f * " + x + " * " + x +
      " * " + x + ")";
  return "0.5f * " + x + " * (1.0f + " +
         msl_stable_tanh_expr(tanh_arg) + ")";
}

} // namespace gfx_plugin
} // namespace ov
