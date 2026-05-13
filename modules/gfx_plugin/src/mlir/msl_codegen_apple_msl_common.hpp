// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
namespace gfx_plugin {

ov::Shape static_shape_or_placeholder(const ov::PartialShape &pshape);
std::vector<int64_t> to_i64_shape(const ov::Shape &shape);
std::vector<int64_t> make_strides(const ov::Shape &shape);
void fill_broadcast_strides(const ov::Shape &output_shape,
                            const ov::Shape &input_shape,
                            std::vector<int64_t> &strides);
ov::Shape shape_from_entry_argument(mlir::ModuleOp module, size_t arg_idx,
                                    const ov::Shape &fallback);
ov::Shape
shape_from_entry_argument_or_partial(mlir::ModuleOp module, size_t arg_idx,
                                     const ov::PartialShape &fallback);
ov::Shape output_shape_for_codegen(mlir::ModuleOp module,
                                   const std::shared_ptr<const ov::Node> &node);
std::vector<int64_t> read_absorbed_input_permutation(mlir::ModuleOp module,
                                                     size_t input_idx);
std::string msl_stable_tanh_expr(const std::string &x);
std::string msl_stable_gelu_tanh_expr(const std::string &x);
} // namespace gfx_plugin
} // namespace ov
