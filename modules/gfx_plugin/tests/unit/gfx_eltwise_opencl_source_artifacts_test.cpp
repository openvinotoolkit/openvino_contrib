// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gfx_opencl_source_artifact_verifier.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

using test::OpenClSourceArtifactVerifier;

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

std::shared_ptr<ov::op::v0::Constant> i32_const(ov::Shape shape,
                                                std::vector<int32_t> values) {
  return ov::op::v0::Constant::create(ov::element::i32, std::move(shape),
                                      std::move(values));
}

std::vector<GfxOpenClSourceScalarArg> op_and_broadcast_scalar_args() {
  std::vector<GfxOpenClSourceScalarArg> args = {
      GfxOpenClSourceScalarArg::ElementCount, GfxOpenClSourceScalarArg::OpCode};
  args.insert(args.end(), 13, GfxOpenClSourceScalarArg::StaticU32);
  return args;
}

std::vector<uint32_t> rhs_31_broadcast_to_234_strides() {
  return {
      3, 2, 3, 4, 1, 12, 4, 1, 0, 0, 1, 0, 0,
  };
}

std::vector<uint32_t> lhs_31_broadcast_to_234_strides() {
  return {
      3, 2, 3, 4, 1, 0, 1, 0, 0, 12, 4, 1, 0,
  };
}

std::vector<GfxOpenClSourceScalarArg> scalar_input_args() {
  return {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::OpCode,
      GfxOpenClSourceScalarArg::InputMode,
  };
}

std::vector<GfxOpenClSourceScalarArg> scalar_constant_args() {
  return {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::OpCode,
      GfxOpenClSourceScalarArg::InputMode,
      GfxOpenClSourceScalarArg::ScalarConstantF32,
  };
}

} // namespace

TEST(EltwiseOpenClSourceArtifactsTest,
     SameShapeBinaryArtifactsUseSharedOpenClManifest) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto rhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto multiply = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);

  OpenClSourceArtifactVerifier(multiply)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_binary_f32",
                       "gfx_opencl_generated_eltwise_binary_f32", 5u, 2u)
      .excludes({"long", "gfx_opencl_baseline_binary_broadcast_f32",
                 "gfx_opencl_baseline_binary_scalar_f32",
                 "gfx_opencl_baseline_binary_const_f32",
                 "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_select_f32"})
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     TypedBinaryArtifactsUseSharedOpenClManifest) {
  struct Case {
    std::string name;
    std::shared_ptr<ov::Node> node;
    std::string suffix;
    GfxOpenClArtifactOp op;
  };

  const auto f16_lhs = param(ov::element::f16, ov::Shape{2, 3, 4});
  const auto f16_rhs = param(ov::element::f16, ov::Shape{2, 3, 4});
  const auto i32_lhs = param(ov::element::i32, ov::Shape{2, 3, 4});
  const auto i32_rhs = param(ov::element::i32, ov::Shape{2, 3, 4});

  const std::vector<Case> cases = {
      {"f16 SquaredDifference",
       std::make_shared<ov::op::v0::SquaredDifference>(f16_lhs, f16_rhs), "f16",
       GfxOpenClArtifactOp::SquaredDifference},
      {"i32 Divide", std::make_shared<ov::op::v1::Divide>(i32_lhs, i32_rhs),
       "i32", GfxOpenClArtifactOp::Divide},
      {"i32 Mod", std::make_shared<ov::op::v1::Mod>(i32_lhs, i32_rhs), "i32",
       GfxOpenClArtifactOp::Mod},
      {"i32 FloorMod", std::make_shared<ov::op::v1::FloorMod>(i32_lhs, i32_rhs),
       "i32", GfxOpenClArtifactOp::FloorMod},
      {"i32 Power", std::make_shared<ov::op::v1::Power>(i32_lhs, i32_rhs),
       "i32", GfxOpenClArtifactOp::Power},
  };

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    auto verifier =
        OpenClSourceArtifactVerifier(test_case.node)
            .expect_artifact(
                GfxKernelStageFamily::Eltwise,
                "opencl/generated/eltwise_binary_" + test_case.suffix,
                "gfx_opencl_generated_eltwise_binary_" + test_case.suffix, 5u,
                2u)
            .excludes({"long", "__global const long*",
                       "gfx_opencl_baseline_binary_f32",
                       "gfx_opencl_baseline_binary_broadcast_f32",
                       "gfx_opencl_baseline_binary_scalar_f32",
                       "gfx_opencl_baseline_binary_const_f32"})
            .has_op(test_case.op)
            .supports_opencl_compiler();
    if (test_case.name == "i32 Power") {
      verifier.contains_source("gfx_pow_i32_exact")
          .excludes({"(int)pow((float)lhs, (float)rhs)"});
    }
  }
}

TEST(EltwiseOpenClSourceArtifactsTest,
     TypedScalarBinaryArtifactsKeepScalarInputsAsTensorSlots) {
  const auto f16_tensor = param(ov::element::f16, ov::Shape{2, 3, 4});
  const auto f16_scalar = f16_const(ov::Shape{}, {2.0f});
  const auto i32_tensor = param(ov::element::i32, ov::Shape{2, 3, 4});
  const auto i32_scalar = i32_const(ov::Shape{}, {5});

  const auto f16_multiply =
      std::make_shared<ov::op::v1::Multiply>(f16_tensor, f16_scalar);
  OpenClSourceArtifactVerifier(f16_multiply)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_scalar_f16",
                       "gfx_opencl_generated_eltwise_scalar_f16", 6u, 2u,
                       scalar_input_args(), {0, 1})
      .excludes({"long", "__global const long*",
                 "gfx_opencl_baseline_binary_const_f32"})
      .has_input_mode(GfxOpenClArtifactInputMode::RhsScalar)
      .supports_opencl_compiler();

  const auto i32_floor_mod =
      std::make_shared<ov::op::v1::FloorMod>(i32_scalar, i32_tensor);
  OpenClSourceArtifactVerifier(i32_floor_mod)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_scalar_i32",
                       "gfx_opencl_generated_eltwise_scalar_i32", 6u, 2u,
                       scalar_input_args(), {0, 1})
      .excludes({"long", "__global const long*",
                 "gfx_opencl_baseline_binary_const_f32"})
      .has_input_mode(GfxOpenClArtifactInputMode::LhsScalar)
      .has_op(GfxOpenClArtifactOp::FloorMod)
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     TypedBroadcastBinaryArtifactsCarryAlignedStrideMetadata) {
  const auto i32_lhs = param(ov::element::i32, ov::Shape{2, 3, 4});
  const auto i32_rhs = param(ov::element::i32, ov::Shape{3, 1});
  const auto i32_mod = std::make_shared<ov::op::v1::Mod>(i32_lhs, i32_rhs);
  OpenClSourceArtifactVerifier(i32_mod)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_broadcast_i32",
                       "gfx_opencl_generated_eltwise_broadcast_i32", 18u, 2u,
                       op_and_broadcast_scalar_args(), {0, 1},
                       rhs_31_broadcast_to_234_strides())
      .excludes({"long", "__global const long*",
                 "gfx_opencl_baseline_binary_broadcast_f32"})
      .has_op(GfxOpenClArtifactOp::Mod)
      .supports_opencl_compiler();

  const auto f16_lhs = param(ov::element::f16, ov::Shape{3, 1});
  const auto f16_rhs = param(ov::element::f16, ov::Shape{2, 3, 4});
  const auto f16_sub = std::make_shared<ov::op::v1::Subtract>(f16_lhs, f16_rhs);
  OpenClSourceArtifactVerifier(f16_sub)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_broadcast_f16",
                       "gfx_opencl_generated_eltwise_broadcast_f16", 18u, 2u,
                       op_and_broadcast_scalar_args(), {0, 1},
                       lhs_31_broadcast_to_234_strides())
      .excludes({"long", "__global const long*",
                 "gfx_opencl_baseline_binary_broadcast_f32"})
      .has_op(GfxOpenClArtifactOp::Subtract)
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     ConstantVectorBinaryArtifactsKeepConstantInputsAsTensorSlots) {
  const auto tensor = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto same_shape_const =
      f32_const(ov::Shape{2, 3, 4}, std::vector<float>(24, 2.0f));
  const auto broadcast_const =
      f32_const(ov::Shape{3, 1}, std::vector<float>(3, 2.0f));
  const auto multiply_same_shape =
      std::make_shared<ov::op::v1::Multiply>(tensor, same_shape_const);
  const auto multiply_broadcast =
      std::make_shared<ov::op::v1::Multiply>(tensor, broadcast_const);

  OpenClSourceArtifactVerifier(multiply_same_shape)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_binary_f32",
                       "gfx_opencl_generated_eltwise_binary_f32", 5u, 2u,
                       {GfxOpenClSourceScalarArg::ElementCount,
                        GfxOpenClSourceScalarArg::OpCode},
                       {0, 1})
      .supports_opencl_compiler();

  OpenClSourceArtifactVerifier(multiply_broadcast)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_broadcast_f32",
                       "gfx_opencl_generated_eltwise_broadcast_f32", 18u, 2u,
                       op_and_broadcast_scalar_args(), {0, 1},
                       rhs_31_broadcast_to_234_strides())
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     BroadcastBinaryArtifactsCarryAlignedStrideMetadata) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto rhs = param(ov::element::f32, ov::Shape{3, 1});
  const auto multiply = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);

  OpenClSourceArtifactVerifier(multiply)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_broadcast_f32",
                       "gfx_opencl_generated_eltwise_broadcast_f32", 18u, 2u,
                       op_and_broadcast_scalar_args(), {0, 1},
                       rhs_31_broadcast_to_234_strides())
      .excludes({"long", "out_dim[4]", "lhs_stride[4]",
                 "gfx_opencl_baseline_binary_scalar_f32",
                 "gfx_opencl_baseline_select_f32"})
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     BroadcastBinaryArtifactsKeepInputSlotsForLhsBroadcast) {
  const auto lhs = param(ov::element::f32, ov::Shape{3, 1});
  const auto rhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto sub = std::make_shared<ov::op::v1::Subtract>(lhs, rhs);

  OpenClSourceArtifactVerifier(sub)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_broadcast_f32",
                       "gfx_opencl_generated_eltwise_broadcast_f32", 18u, 2u,
                       op_and_broadcast_scalar_args(), {0, 1},
                       lhs_31_broadcast_to_234_strides())
      .has_op(GfxOpenClArtifactOp::Subtract)
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     ScalarBinaryArtifactsUseManifestRolesAndInputSlotMetadata) {
  const auto tensor = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto scalar = param(ov::element::f32, ov::Shape{1});
  const auto rhs_scalar =
      std::make_shared<ov::op::v1::Subtract>(tensor, scalar);
  const auto lhs_scalar =
      std::make_shared<ov::op::v1::Subtract>(scalar, tensor);
  const auto rhs_const = std::make_shared<ov::op::v1::Multiply>(
      tensor, f32_const(ov::Shape{}, {2.0f}));
  const auto lhs_const = std::make_shared<ov::op::v1::Subtract>(
      f32_const(ov::Shape{}, {2.0f}), tensor);

  OpenClSourceArtifactVerifier(rhs_scalar)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_scalar_f32",
                       "gfx_opencl_generated_eltwise_scalar_f32", 6u, 2u,
                       scalar_input_args(), {0, 1})
      .excludes({"long", "gfx_opencl_baseline_binary_f32",
                 "gfx_opencl_baseline_binary_broadcast_f32",
                 "gfx_opencl_baseline_binary_const_f32",
                 "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_select_f32"})
      .has_input_mode(GfxOpenClArtifactInputMode::RhsScalar)
      .supports_opencl_compiler();

  OpenClSourceArtifactVerifier(lhs_scalar)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_scalar_f32",
                       "gfx_opencl_generated_eltwise_scalar_f32", 6u, 2u,
                       scalar_input_args(), {0, 1})
      .has_input_mode(GfxOpenClArtifactInputMode::LhsScalar)
      .supports_opencl_compiler();

  OpenClSourceArtifactVerifier(rhs_const)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_const_f32",
                       "gfx_opencl_generated_eltwise_const_f32", 6u, 1u,
                       scalar_constant_args(), {0})
      .excludes({"long", "gfx_opencl_baseline_binary_f32",
                 "gfx_opencl_baseline_binary_broadcast_f32",
                 "gfx_opencl_baseline_binary_scalar_f32",
                 "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_select_f32"})
      .has_input_mode(GfxOpenClArtifactInputMode::RhsScalarConstant)
      .has_scalar_constant(2.0f)
      .supports_opencl_compiler();

  OpenClSourceArtifactVerifier(lhs_const)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_const_f32",
                       "gfx_opencl_generated_eltwise_const_f32", 6u, 1u,
                       scalar_constant_args(), {1})
      .has_input_mode(GfxOpenClArtifactInputMode::LhsScalarConstant)
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     CompareAndSelectArtifactsUseTheSameSourceManifestPath) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3});
  const auto rhs = param(ov::element::f32, ov::Shape{2, 3});
  const auto condition = param(ov::element::boolean, ov::Shape{2, 3});
  const auto greater = std::make_shared<ov::op::v1::Greater>(lhs, rhs);
  const auto select = std::make_shared<ov::op::v1::Select>(condition, lhs, rhs);

  OpenClSourceArtifactVerifier(greater)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/baseline/compare_f32",
                       "gfx_opencl_baseline_compare_f32", 5u, 2u)
      .excludes({"long", "gfx_opencl_baseline_binary_f32",
                 "gfx_opencl_baseline_binary_broadcast_f32",
                 "gfx_opencl_baseline_binary_scalar_f32",
                 "gfx_opencl_baseline_binary_const_f32",
                 "gfx_opencl_baseline_compare_broadcast_f32",
                 "gfx_opencl_baseline_select_f32",
                 "gfx_opencl_baseline_select_broadcast_f32"})
      .has_op(GfxOpenClArtifactOp::Greater)
      .supports_opencl_compiler();

  OpenClSourceArtifactVerifier(select)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/baseline/select_f32",
                       "gfx_opencl_baseline_select_f32", 5u, 3u,
                       {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2})
      .excludes({"long", "gfx_opencl_baseline_binary_f32",
                 "gfx_opencl_baseline_binary_broadcast_f32",
                 "gfx_opencl_baseline_binary_scalar_f32",
                 "gfx_opencl_baseline_binary_const_f32",
                 "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_compare_broadcast_f32",
                 "gfx_opencl_baseline_select_broadcast_f32"})
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     CompareAndSelectBroadcastArtifactsCarryAlignedStrideMetadata) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto rhs = param(ov::element::f32, ov::Shape{3, 1});
  const auto greater = std::make_shared<ov::op::v1::Greater>(lhs, rhs);

  std::vector<GfxOpenClSourceScalarArg> compare_args = {
      GfxOpenClSourceScalarArg::ElementCount, GfxOpenClSourceScalarArg::OpCode};
  compare_args.insert(compare_args.end(), 13,
                      GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> compare_static_u32_scalars = {
      3, 2, 3, 4, 1, 12, 4, 1, 0, 0, 1, 0, 0,
  };

  OpenClSourceArtifactVerifier(greater)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/baseline/compare_broadcast_f32",
                       "gfx_opencl_baseline_compare_broadcast_f32", 18u, 2u,
                       compare_args, {0, 1}, compare_static_u32_scalars)
      .excludes({"long", "gfx_opencl_baseline_binary_f32",
                 "gfx_opencl_baseline_binary_broadcast_f32",
                 "gfx_opencl_baseline_binary_scalar_f32",
                 "gfx_opencl_baseline_binary_const_f32",
                 "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_select_f32",
                 "gfx_opencl_baseline_select_broadcast_f32"})
      .has_op(GfxOpenClArtifactOp::Greater)
      .supports_opencl_compiler();

  const auto cond = param(ov::element::boolean, ov::Shape{1, 3, 1});
  const auto then_data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto else_data = param(ov::element::f32, ov::Shape{1, 1, 4});
  const auto select =
      std::make_shared<ov::op::v1::Select>(cond, then_data, else_data);

  std::vector<GfxOpenClSourceScalarArg> select_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  select_args.insert(select_args.end(), 17,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> select_static_u32_scalars = {
      3, 2, 3, 4, 1, 0, 1, 0, 0, 12, 4, 1, 0, 0, 0, 1, 0,
  };

  OpenClSourceArtifactVerifier(select)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/baseline/select_broadcast_f32",
                       "gfx_opencl_baseline_select_broadcast_f32", 22u, 3u,
                       select_args, {0, 1, 2}, select_static_u32_scalars)
      .excludes({"long", "gfx_opencl_baseline_binary_f32",
                 "gfx_opencl_baseline_binary_broadcast_f32",
                 "gfx_opencl_baseline_binary_scalar_f32",
                 "gfx_opencl_baseline_binary_const_f32",
                 "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_compare_broadcast_f32",
                 "gfx_opencl_baseline_select_f32"})
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     LogicalBoolArtifactsUsePackedBooleanSourceBundles) {
  const auto lhs = param(ov::element::boolean, ov::Shape{2, 3});
  const auto rhs = param(ov::element::boolean, ov::Shape{2, 3});

  const auto logical_not = std::make_shared<ov::op::v1::LogicalNot>(lhs);
  OpenClSourceArtifactVerifier(logical_not)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/baseline/logical_unary_bool",
                       "gfx_opencl_baseline_logical_unary_bool", 4u, 1u)
      .excludes({"float", "long", "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_compare_broadcast_f32",
                 "gfx_opencl_baseline_select_f32",
                 "gfx_opencl_baseline_select_broadcast_f32",
                 "gfx_opencl_baseline_logical_binary_bool",
                 "gfx_opencl_baseline_logical_binary_broadcast_bool"})
      .has_op(GfxOpenClArtifactOp::LogicalNot)
      .supports_opencl_compiler();

  struct BinaryCase {
    std::string name;
    std::shared_ptr<ov::Node> node;
    GfxOpenClArtifactOp op;
  };
  const std::vector<BinaryCase> binary_cases = {
      {"LogicalAnd", std::make_shared<ov::op::v1::LogicalAnd>(lhs, rhs),
       GfxOpenClArtifactOp::LogicalAnd},
      {"LogicalOr", std::make_shared<ov::op::v1::LogicalOr>(lhs, rhs),
       GfxOpenClArtifactOp::LogicalOr},
      {"LogicalXor", std::make_shared<ov::op::v1::LogicalXor>(lhs, rhs),
       GfxOpenClArtifactOp::LogicalXor},
  };

  for (const auto &test_case : binary_cases) {
    SCOPED_TRACE(test_case.name);
    OpenClSourceArtifactVerifier(test_case.node)
        .expect_artifact(GfxKernelStageFamily::Eltwise,
                         "opencl/baseline/logical_binary_bool",
                         "gfx_opencl_baseline_logical_binary_bool", 5u, 2u)
        .excludes({"float", "long", "gfx_opencl_baseline_compare_f32",
                   "gfx_opencl_baseline_compare_broadcast_f32",
                   "gfx_opencl_baseline_select_f32",
                   "gfx_opencl_baseline_select_broadcast_f32",
                   "gfx_opencl_baseline_logical_unary_bool",
                   "gfx_opencl_baseline_logical_binary_broadcast_bool"})
        .has_op(test_case.op)
        .supports_opencl_compiler();
  }
}

TEST(EltwiseOpenClSourceArtifactsTest,
     LogicalBoolBroadcastArtifactsCarryAlignedStrideMetadata) {
  const auto lhs = param(ov::element::boolean, ov::Shape{2, 3, 4});
  const auto rhs = param(ov::element::boolean, ov::Shape{3, 1});
  const auto logical_or = std::make_shared<ov::op::v1::LogicalOr>(lhs, rhs);

  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount, GfxOpenClSourceScalarArg::OpCode};
  scalar_args.insert(scalar_args.end(), 13,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3, 2, 3, 4, 1, 12, 4, 1, 0, 0, 1, 0, 0,
  };

  OpenClSourceArtifactVerifier(logical_or)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/baseline/logical_binary_broadcast_bool",
                       "gfx_opencl_baseline_logical_binary_broadcast_bool", 18u,
                       2u, scalar_args, {0, 1}, static_u32_scalars)
      .excludes({"float", "long", "gfx_opencl_baseline_compare_f32",
                 "gfx_opencl_baseline_compare_broadcast_f32",
                 "gfx_opencl_baseline_select_f32",
                 "gfx_opencl_baseline_select_broadcast_f32",
                 "gfx_opencl_baseline_logical_unary_bool",
                 "gfx_opencl_baseline_logical_binary_bool"})
      .has_op(GfxOpenClArtifactOp::LogicalOr)
      .supports_opencl_compiler();
}

TEST(EltwiseOpenClSourceArtifactsTest,
     DynamicF16SelectUsesSameShapeRuntimeElementCount) {
  const auto cond = std::make_shared<ov::op::v0::Parameter>(
      ov::element::boolean, ov::PartialShape{1, -1, 4});
  const auto then_data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  const auto else_data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  const auto select =
      std::make_shared<ov::op::v1::Select>(cond, then_data, else_data);

  OpenClSourceArtifactVerifier(select)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/baseline/select_f16_dynamic",
                       "gfx_opencl_baseline_select_f16", 5u, 3u,
                       {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2})
      .supports_opencl_compiler();
}

} // namespace gfx_plugin
} // namespace ov
