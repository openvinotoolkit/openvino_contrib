// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_eltwise_contract_cases.hpp"

#include <utility>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

std::shared_ptr<ov::op::v0::Parameter> param(const ov::element::Type &type,
                                             ov::Shape shape) {
  return std::make_shared<ov::op::v0::Parameter>(type, std::move(shape));
}

std::shared_ptr<ov::op::v0::Constant> f32_const(ov::Shape shape,
                                                std::vector<float> values) {
  return ov::op::v0::Constant::create(ov::element::f32, std::move(shape),
                                      std::move(values));
}

std::shared_ptr<ov::op::v0::Constant> f16_const(ov::Shape shape,
                                                std::vector<float> values) {
  return ov::op::v0::Constant::create(ov::element::f16, std::move(shape),
                                      std::move(values));
}

std::vector<uint32_t> broadcast_234_by_31_scalars() {
  return {
      3, 2, 3, 4, 1, 12, 4, 1, 0, 0, 1, 0, 0,
  };
}

std::vector<GfxOpenClSourceScalarArg> broadcast_scalar_args() {
  std::vector<GfxOpenClSourceScalarArg> args = {
      GfxOpenClSourceScalarArg::ElementCount, GfxOpenClSourceScalarArg::OpCode};
  args.insert(args.end(), 13, GfxOpenClSourceScalarArg::StaticU32);
  return args;
}

} // namespace

std::vector<EltwiseOpenClArtifactCase> eltwise_opencl_artifact_cases() {
  return {
      {"F32SameShapeAdd",
       [] {
         return std::make_shared<ov::op::v1::Add>(
             param(ov::element::f32, ov::Shape{2, 3, 4}),
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/eltwise_binary_f32",
       "gfx_opencl_generated_eltwise_binary_f32",
       GfxOpenClArtifactOp::Add,
       5u,
       2u,
       {0, 1},
       {GfxOpenClSourceScalarArg::ElementCount,
        GfxOpenClSourceScalarArg::OpCode},
       {},
       GfxOpenClArtifactInputMode::Direct},
      {"F32RhsConstSubtract",
       [] {
         return std::make_shared<ov::op::v1::Subtract>(
             param(ov::element::f32, ov::Shape{2, 3, 4}),
             f32_const(ov::Shape{}, {2.0f}));
       },
       "opencl/generated/eltwise_const_f32",
       "gfx_opencl_generated_eltwise_const_f32",
       GfxOpenClArtifactOp::Subtract,
       6u,
       1u,
       {0},
       {GfxOpenClSourceScalarArg::ElementCount,
        GfxOpenClSourceScalarArg::OpCode, GfxOpenClSourceScalarArg::InputMode,
        GfxOpenClSourceScalarArg::ScalarConstantF32},
       {},
       GfxOpenClArtifactInputMode::RhsScalarConstant},
      {"F16RhsScalarMultiply",
       [] {
         return std::make_shared<ov::op::v1::Multiply>(
             param(ov::element::f16, ov::Shape{2, 3, 4}),
             f16_const(ov::Shape{}, {2.0f}));
       },
       "opencl/generated/eltwise_scalar_f16",
       "gfx_opencl_generated_eltwise_scalar_f16",
       GfxOpenClArtifactOp::Multiply,
       6u,
       2u,
       {0, 1},
       {GfxOpenClSourceScalarArg::ElementCount,
        GfxOpenClSourceScalarArg::OpCode, GfxOpenClSourceScalarArg::InputMode},
       {},
       GfxOpenClArtifactInputMode::RhsScalar},
      {"I32NumpyBroadcastMod",
       [] {
         return std::make_shared<ov::op::v1::Mod>(
             param(ov::element::i32, ov::Shape{2, 3, 4}),
             param(ov::element::i32, ov::Shape{3, 1}));
       },
       "opencl/generated/eltwise_broadcast_i32",
       "gfx_opencl_generated_eltwise_broadcast_i32",
       GfxOpenClArtifactOp::Mod,
       18u,
       2u,
       {0, 1},
       broadcast_scalar_args(),
       broadcast_234_by_31_scalars(),
       GfxOpenClArtifactInputMode::Direct},
      {"F32CompareGreater",
       [] {
         return std::make_shared<ov::op::v1::Greater>(
             param(ov::element::f32, ov::Shape{2, 3}),
             param(ov::element::f32, ov::Shape{2, 3}));
       },
       "opencl/generated/eltwise_compare_f32",
       "gfx_opencl_generated_eltwise_compare_f32",
       GfxOpenClArtifactOp::Greater,
       5u,
       2u,
       {0, 1},
       {GfxOpenClSourceScalarArg::ElementCount,
        GfxOpenClSourceScalarArg::OpCode},
       {},
       GfxOpenClArtifactInputMode::Direct},
      {"F32SelectSameShape",
       [] {
         auto cond = param(ov::element::boolean, ov::Shape{2, 3});
         auto lhs = param(ov::element::f32, ov::Shape{2, 3});
         auto rhs = param(ov::element::f32, ov::Shape{2, 3});
         return std::make_shared<ov::op::v1::Select>(cond, lhs, rhs);
       },
       "opencl/generated/eltwise_select_f32",
       "gfx_opencl_generated_eltwise_select_f32",
       GfxOpenClArtifactOp::Identity,
       5u,
       3u,
       {0, 1, 2},
       {GfxOpenClSourceScalarArg::ElementCount},
       {},
       GfxOpenClArtifactInputMode::Direct},
      {"BoolLogicalNot",
       [] {
         return std::make_shared<ov::op::v1::LogicalNot>(
             param(ov::element::boolean, ov::Shape{2, 3}));
       },
       "opencl/generated/eltwise_logical_unary_bool",
       "gfx_opencl_generated_eltwise_logical_unary_bool",
       GfxOpenClArtifactOp::LogicalNot,
       4u,
       1u,
       {0},
       {GfxOpenClSourceScalarArg::ElementCount,
        GfxOpenClSourceScalarArg::OpCode},
       {},
       GfxOpenClArtifactInputMode::Direct},
      {"BoolLogicalAnd",
       [] {
         return std::make_shared<ov::op::v1::LogicalAnd>(
             param(ov::element::boolean, ov::Shape{2, 3}),
             param(ov::element::boolean, ov::Shape{2, 3}));
       },
       "opencl/generated/eltwise_logical_binary_bool",
       "gfx_opencl_generated_eltwise_logical_binary_bool",
       GfxOpenClArtifactOp::LogicalAnd,
       5u,
       2u,
       {0, 1},
       {GfxOpenClSourceScalarArg::ElementCount,
        GfxOpenClSourceScalarArg::OpCode},
       {},
       GfxOpenClArtifactInputMode::Direct},
  };
}

} // namespace gfx_plugin
} // namespace ov
