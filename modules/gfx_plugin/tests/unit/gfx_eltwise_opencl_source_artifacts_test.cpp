// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backends/opencl/compiler/opencl_eltwise_kernel_unit.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "compiler/kernel_registry.hpp"
#include "compiler/lowering_planner.hpp"
#include "compiler/operation_support.hpp"
#include "gfx_opencl_source_artifact_verifier.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "kernel_ir/opencl_kernels/eltwise_kernel.hpp"
#include "openvino/core/model.hpp"
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
#include "openvino/op/result.hpp"
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

void expect_opencl_eltwise_kernel_unit_owner(
    const std::shared_ptr<const ov::Node> &node,
    const std::string &expected_source_id) {
  const auto artifact =
      compiler::make_opencl_eltwise_source_artifact(node, expected_source_id);
  ASSERT_TRUE(artifact.has_value());
  ASSERT_TRUE(artifact->valid);
  EXPECT_EQ(artifact->stage_manifest.stage_family,
            GfxKernelStageFamily::Eltwise);
  EXPECT_EQ(artifact->artifact_ref.source_id, expected_source_id);

  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const auto registry = compiler::make_opencl_kernel_registry(target);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy(registry));
  const auto support = capabilities.query_operation({node});
  ASSERT_TRUE(support.semantic_legal) << support.semantic_reason;
  EXPECT_EQ(support.semantic_reason, "registered_opencl_eltwise_kernel_unit");
  EXPECT_EQ(support.preferred_route_kind,
            compiler::LoweringRouteKind::GeneratedKernel);
  EXPECT_EQ(support.preferred_route, expected_source_id);
}

std::vector<GfxOpenClSourceScalarArg> scalar_constant_args() {
  return {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::OpCode,
      GfxOpenClSourceScalarArg::InputMode,
      GfxOpenClSourceScalarArg::ScalarConstantF32,
  };
}

void expect_generated_opencl_kernel_unit(
    const std::shared_ptr<ov::Node> &node,
    const std::string &expected_kernel_unit_id,
    ov::ParameterVector parameters) {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy());
  const auto support = capabilities.query_operation({node});
  ASSERT_TRUE(support.semantic_legal);
  EXPECT_EQ(support.preferred_route_kind,
            compiler::LoweringRouteKind::GeneratedKernel);
  EXPECT_EQ(support.preferred_route, expected_kernel_unit_id);

  const auto registry = compiler::make_opencl_kernel_registry(target);
  const auto unit = registry.resolve(
      compiler::LoweringRouteKind::GeneratedKernel, expected_kernel_unit_id);
  ASSERT_TRUE(unit.valid());
  EXPECT_EQ(unit.kind(), compiler::KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(unit.backend_domain(), "opencl");
  EXPECT_EQ(unit.op_family(), "Eltwise");

  compiler::OperationLegalizer legalizer(capabilities);
  compiler::LoweringPlanner planner(target, registry);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{std::make_shared<ov::op::v0::Result>(node)},
      std::move(parameters));
  const auto plan = planner.plan(model, legalizer);
  ASSERT_TRUE(plan.executable());

  bool found = false;
  for (const auto &operation : plan.operations) {
    if (operation.kernel_unit.id() == expected_kernel_unit_id) {
      found = true;
      EXPECT_EQ(operation.kernel_unit.kind(),
                compiler::KernelUnitKind::GeneratedKernel);
      EXPECT_EQ(operation.kernel_unit.route_kind(),
                compiler::LoweringRouteKind::GeneratedKernel);
    }
  }
  EXPECT_TRUE(found);
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
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      multiply, "opencl/generated/eltwise_binary_f32", {lhs, rhs});
}

TEST(EltwiseOpenClSourceArtifactsTest,
     TypedBinaryArtifactsUseSharedOpenClManifest) {
  struct Case {
    std::string name;
    std::shared_ptr<ov::Node> node;
    std::string suffix;
    GfxOpenClArtifactOp op;
    ov::ParameterVector parameters;
  };

  const auto f16_lhs = param(ov::element::f16, ov::Shape{2, 3, 4});
  const auto f16_rhs = param(ov::element::f16, ov::Shape{2, 3, 4});
  const auto i32_lhs = param(ov::element::i32, ov::Shape{2, 3, 4});
  const auto i32_rhs = param(ov::element::i32, ov::Shape{2, 3, 4});

  const std::vector<Case> cases = {
      {"f16 SquaredDifference",
       std::make_shared<ov::op::v0::SquaredDifference>(f16_lhs, f16_rhs),
       "f16",
       GfxOpenClArtifactOp::SquaredDifference,
       {f16_lhs, f16_rhs}},
      {"i32 Divide",
       std::make_shared<ov::op::v1::Divide>(i32_lhs, i32_rhs),
       "i32",
       GfxOpenClArtifactOp::Divide,
       {i32_lhs, i32_rhs}},
      {"i32 Mod",
       std::make_shared<ov::op::v1::Mod>(i32_lhs, i32_rhs),
       "i32",
       GfxOpenClArtifactOp::Mod,
       {i32_lhs, i32_rhs}},
      {"i32 FloorMod",
       std::make_shared<ov::op::v1::FloorMod>(i32_lhs, i32_rhs),
       "i32",
       GfxOpenClArtifactOp::FloorMod,
       {i32_lhs, i32_rhs}},
      {"i32 Power",
       std::make_shared<ov::op::v1::Power>(i32_lhs, i32_rhs),
       "i32",
       GfxOpenClArtifactOp::Power,
       {i32_lhs, i32_rhs}},
  };

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    OpenClSourceArtifactVerifier(test_case.node)
        .expect_artifact(
            GfxKernelStageFamily::Eltwise,
            "opencl/generated/eltwise_binary_" + test_case.suffix,
            "gfx_opencl_generated_eltwise_binary_" + test_case.suffix, 5u, 2u)
        .has_op(test_case.op)
        .supports_opencl_compiler();
    expect_generated_opencl_kernel_unit(
        test_case.node, "opencl/generated/eltwise_binary_" + test_case.suffix,
        test_case.parameters);
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
      .has_input_mode(GfxOpenClArtifactInputMode::RhsScalar)
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      f16_multiply, "opencl/generated/eltwise_scalar_f16", {f16_tensor});

  const auto i32_floor_mod =
      std::make_shared<ov::op::v1::FloorMod>(i32_scalar, i32_tensor);
  OpenClSourceArtifactVerifier(i32_floor_mod)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_scalar_i32",
                       "gfx_opencl_generated_eltwise_scalar_i32", 6u, 2u,
                       scalar_input_args(), {0, 1})
      .has_input_mode(GfxOpenClArtifactInputMode::LhsScalar)
      .has_op(GfxOpenClArtifactOp::FloorMod)
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      i32_floor_mod, "opencl/generated/eltwise_scalar_i32", {i32_tensor});
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
      .has_op(GfxOpenClArtifactOp::Mod)
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      i32_mod, "opencl/generated/eltwise_broadcast_i32", {i32_lhs, i32_rhs});

  const auto f16_lhs = param(ov::element::f16, ov::Shape{3, 1});
  const auto f16_rhs = param(ov::element::f16, ov::Shape{2, 3, 4});
  const auto f16_sub = std::make_shared<ov::op::v1::Subtract>(f16_lhs, f16_rhs);
  OpenClSourceArtifactVerifier(f16_sub)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_broadcast_f16",
                       "gfx_opencl_generated_eltwise_broadcast_f16", 18u, 2u,
                       op_and_broadcast_scalar_args(), {0, 1},
                       lhs_31_broadcast_to_234_strides())
      .has_op(GfxOpenClArtifactOp::Subtract)
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      f16_sub, "opencl/generated/eltwise_broadcast_f16", {f16_lhs, f16_rhs});
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
  expect_generated_opencl_kernel_unit(
      multiply_same_shape, "opencl/generated/eltwise_binary_f32", {tensor});

  OpenClSourceArtifactVerifier(multiply_broadcast)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_broadcast_f32",
                       "gfx_opencl_generated_eltwise_broadcast_f32", 18u, 2u,
                       op_and_broadcast_scalar_args(), {0, 1},
                       rhs_31_broadcast_to_234_strides())
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      multiply_broadcast, "opencl/generated/eltwise_broadcast_f32", {tensor});
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
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      multiply, "opencl/generated/eltwise_broadcast_f32", {lhs, rhs});
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
  expect_generated_opencl_kernel_unit(
      sub, "opencl/generated/eltwise_broadcast_f32", {lhs, rhs});
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
      .has_input_mode(GfxOpenClArtifactInputMode::RhsScalar)
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      rhs_scalar, "opencl/generated/eltwise_scalar_f32", {tensor, scalar});

  OpenClSourceArtifactVerifier(lhs_scalar)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_scalar_f32",
                       "gfx_opencl_generated_eltwise_scalar_f32", 6u, 2u,
                       scalar_input_args(), {0, 1})
      .has_input_mode(GfxOpenClArtifactInputMode::LhsScalar)
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      lhs_scalar, "opencl/generated/eltwise_scalar_f32", {tensor, scalar});

  OpenClSourceArtifactVerifier(rhs_const)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_const_f32",
                       "gfx_opencl_generated_eltwise_const_f32", 6u, 1u,
                       scalar_constant_args(), {0})
      .has_input_mode(GfxOpenClArtifactInputMode::RhsScalarConstant)
      .has_scalar_constant(2.0f)
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      rhs_const, "opencl/generated/eltwise_const_f32", {tensor});

  OpenClSourceArtifactVerifier(lhs_const)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_const_f32",
                       "gfx_opencl_generated_eltwise_const_f32", 6u, 1u,
                       scalar_constant_args(), {1})
      .has_input_mode(GfxOpenClArtifactInputMode::LhsScalarConstant)
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      lhs_const, "opencl/generated/eltwise_const_f32", {tensor});
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
                       "opencl/generated/eltwise_compare_f32",
                       "gfx_opencl_generated_eltwise_compare_f32", 5u, 2u)
      .has_op(GfxOpenClArtifactOp::Greater)
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      greater, "opencl/generated/eltwise_compare_f32", {lhs, rhs});
  expect_opencl_eltwise_kernel_unit_owner(
      greater, "opencl/generated/eltwise_compare_f32");

  OpenClSourceArtifactVerifier(select)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_select_f32",
                       "gfx_opencl_generated_eltwise_select_f32", 5u, 3u,
                       {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2})
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      select, "opencl/generated/eltwise_select_f32", {condition, lhs, rhs});
  expect_opencl_eltwise_kernel_unit_owner(
      select, "opencl/generated/eltwise_select_f32");
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
                       "opencl/generated/eltwise_compare_broadcast_f32",
                       "gfx_opencl_generated_eltwise_compare_broadcast_f32",
                       18u, 2u, compare_args, {0, 1},
                       compare_static_u32_scalars)
      .has_op(GfxOpenClArtifactOp::Greater)
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      greater, "opencl/generated/eltwise_compare_broadcast_f32", {lhs, rhs});

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
                       "opencl/generated/eltwise_select_broadcast_f32",
                       "gfx_opencl_generated_eltwise_select_broadcast_f32", 22u,
                       3u, select_args, {0, 1, 2}, select_static_u32_scalars)
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      select, "opencl/generated/eltwise_select_broadcast_f32",
      {cond, then_data, else_data});
}

TEST(EltwiseOpenClSourceArtifactsTest,
     LogicalBoolArtifactsUsePackedBooleanSourceBundles) {
  const auto lhs = param(ov::element::boolean, ov::Shape{2, 3});
  const auto rhs = param(ov::element::boolean, ov::Shape{2, 3});

  const auto logical_not = std::make_shared<ov::op::v1::LogicalNot>(lhs);
  OpenClSourceArtifactVerifier(logical_not)
      .expect_artifact(GfxKernelStageFamily::Eltwise,
                       "opencl/generated/eltwise_logical_unary_bool",
                       "gfx_opencl_generated_eltwise_logical_unary_bool", 4u,
                       1u)
      .has_op(GfxOpenClArtifactOp::LogicalNot)
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      logical_not, "opencl/generated/eltwise_logical_unary_bool", {lhs});
  expect_opencl_eltwise_kernel_unit_owner(
      logical_not, "opencl/generated/eltwise_logical_unary_bool");

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
                         "opencl/generated/eltwise_logical_binary_bool",
                         "gfx_opencl_generated_eltwise_logical_binary_bool", 5u,
                         2u)
        .has_op(test_case.op)
        .supports_opencl_compiler();
    expect_generated_opencl_kernel_unit(
        test_case.node, "opencl/generated/eltwise_logical_binary_bool",
        {lhs, rhs});
    expect_opencl_eltwise_kernel_unit_owner(
        test_case.node, "opencl/generated/eltwise_logical_binary_bool");
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
      .expect_artifact(
          GfxKernelStageFamily::Eltwise,
          "opencl/generated/eltwise_logical_binary_broadcast_bool",
          "gfx_opencl_generated_eltwise_logical_binary_broadcast_bool", 18u, 2u,
          scalar_args, {0, 1}, static_u32_scalars)
      .has_op(GfxOpenClArtifactOp::LogicalOr)
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      logical_or, "opencl/generated/eltwise_logical_binary_broadcast_bool",
      {lhs, rhs});
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
                       "opencl/generated/eltwise_select_f16_dynamic",
                       "gfx_opencl_generated_eltwise_select_f16_dynamic", 5u,
                       3u, {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2})
      .supports_opencl_compiler();
  expect_generated_opencl_kernel_unit(
      select, "opencl/generated/eltwise_select_f16_dynamic",
      {cond, then_data, else_data});
}

} // namespace gfx_plugin
} // namespace ov
