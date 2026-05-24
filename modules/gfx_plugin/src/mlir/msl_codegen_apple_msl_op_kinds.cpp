// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_op_kinds.hpp"

#include <string>

namespace ov {
namespace gfx_plugin {

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
  if (type == "ReduceLogicalAnd")
    return ReduceKind::LogicalAnd;
  if (type == "ReduceLogicalOr")
    return ReduceKind::LogicalOr;
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
  if (type == "LogicalNot")
    return ActivationKind::LogicalNot;
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

} // namespace gfx_plugin
} // namespace ov
