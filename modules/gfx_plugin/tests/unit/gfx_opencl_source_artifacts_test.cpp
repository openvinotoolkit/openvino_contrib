// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_opencl_source_artifacts_contract.hpp"

TEST(GfxOpenClSourceArtifactsTest, BackendTargetIsStableAndCapabilityDriven) {
  const auto target = BackendTarget::from_backend(GpuBackend::OpenCL);
  EXPECT_EQ(target.backend(), GpuBackend::OpenCL);
  EXPECT_NE(target.fingerprint().find("backend=opencl"), std::string::npos);
  EXPECT_TRUE(target.is_compatible_with_fingerprint(target.fingerprint()));

  const auto kernel_registry = make_opencl_kernel_registry(target);
  BackendCapabilities capabilities(
      target, make_opencl_operation_support_policy(kernel_registry));
  const auto audit = kernel_registry.audit();
  ASSERT_TRUE(audit.valid());
  EXPECT_EQ(audit.handwritten_exception_count, 0u);
  EXPECT_EQ(kernel_registry.route_count(
                LoweringRouteKind::HandwrittenKernelException),
            0u);
  OperationLegalizer legalizer(capabilities);
  LoweringPlanner planner(target, kernel_registry);
  auto input = param(ov::element::f32, ov::Shape{2, 3, 4});
  auto shapeof =
      std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
  const auto support = capabilities.query_operation({shapeof});
  EXPECT_TRUE(support.semantic_legal);
  EXPECT_EQ(support.preferred_route_kind, LoweringRouteKind::GeneratedKernel);
  EXPECT_EQ(support.preferred_route, "opencl/generated/shapeof_i64");

  auto result = std::make_shared<ov::op::v0::Result>(shapeof);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                           ov::ParameterVector{input});
  const auto plan = planner.plan(model, legalizer);
  EXPECT_TRUE(plan.executable());
  EXPECT_EQ(plan.route_count(LoweringRouteKind::GeneratedKernel), 1u);
  bool found_generated_kernel_unit = false;
  for (const auto &op : plan.operations) {
    if (op.kernel_unit.route_kind() == LoweringRouteKind::GeneratedKernel) {
      found_generated_kernel_unit = true;
      EXPECT_TRUE(op.kernel_unit.valid());
      EXPECT_EQ(op.kernel_unit.kind(), KernelUnitKind::GeneratedKernel);
      EXPECT_EQ(op.kernel_unit.backend_domain(), target.backend_id());
      EXPECT_EQ(op.kernel_unit.id(), "opencl/generated/shapeof_i64");
    }
  }
  EXPECT_TRUE(found_generated_kernel_unit);

  const auto manifest = ManifestBuilder{}.build(plan);
  EXPECT_TRUE(manifest.verify().valid());
  EXPECT_EQ(manifest.schema_version, 2u);
  EXPECT_EQ(manifest.route_count(LoweringRouteKind::GeneratedKernel), 1u);
  for (const auto &stage : manifest.stages) {
    EXPECT_FALSE(stage.memory.hidden_host_copy_allowed);
    EXPECT_EQ(stage.dispatch.execution_kind, stage.execution_kind);
    EXPECT_EQ(stage.dispatch.backend_domain, stage.backend_domain);
    EXPECT_EQ(stage.dispatch.kernel_unit_id, stage.kernel_unit_id);
    EXPECT_EQ(stage.dispatch.kernel_unit_kind, stage.kernel_unit_kind);
    if (stage.execution_kind == LoweringRouteKind::GeneratedKernel) {
      EXPECT_EQ(stage.kernel_unit_kind, "generated_kernel");
    }
  }

  auto lhs = param(ov::element::f32, ov::Shape{2, 3});
  auto rhs = param(ov::element::f32, ov::Shape{3, 4});
  auto matmul = std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);
  const auto matmul_support = capabilities.query_operation({matmul});
  EXPECT_FALSE(matmul_support.semantic_legal);
  EXPECT_EQ(matmul_support.semantic_reason, "missing_opencl_matmul_kernel_unit");
}


TEST(GfxOpenClSourceArtifactsTest,
     MissingOpenClArtifactsRejectEvenWhenMlirCoversThem) {
  const auto f32 = param(ov::element::f32, ov::Shape{2, 3});
  const auto i32 = param(ov::element::i32, ov::Shape{2, 3});
  const auto high_rank_lhs = param(ov::element::f32, ov::Shape{1, 1, 1, 1, 3});
  const auto high_rank_rhs = param(ov::element::f32, ov::Shape{3});

  const auto i32_abs = std::make_shared<ov::op::v0::Abs>(i32);
  const auto high_rank_broadcast_add =
      std::make_shared<ov::op::v1::Add>(high_rank_lhs, high_rank_rhs);
  const auto convert_to_f16 =
      std::make_shared<ov::op::v0::Convert>(f32, ov::element::f16);

  EXPECT_FALSE(resolve_opencl_catalog_source_artifact(i32_abs).has_value());
  EXPECT_FALSE(
      resolve_opencl_catalog_source_artifact(high_rank_broadcast_add)
          .has_value());
  EXPECT_FALSE(
      resolve_opencl_catalog_source_artifact(convert_to_f16).has_value());

  EXPECT_TRUE(mlir_supports_node(i32_abs));
  EXPECT_TRUE(mlir_supports_node(high_rank_broadcast_add));
  EXPECT_TRUE(mlir_supports_node(convert_to_f16));
  EXPECT_FALSE(opencl_compiler_supports_node(i32_abs));
  EXPECT_FALSE(opencl_compiler_supports_node(high_rank_broadcast_add));
  EXPECT_FALSE(opencl_compiler_supports_node(convert_to_f16));
}
