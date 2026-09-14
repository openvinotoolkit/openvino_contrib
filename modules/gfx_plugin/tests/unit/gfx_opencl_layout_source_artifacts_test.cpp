// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_opencl_source_artifacts_contract.hpp"

TEST(GfxOpenClLayoutSourceArtifactsTest,
     ViewOnlyLayoutOpsCompileAsMetadataWithoutSourceArtifact) {
  const auto data = param(ov::element::f32, ov::Shape{1, 2, 3});
  const auto reshape = std::make_shared<ov::op::v1::Reshape>(
      data, i64_const(ov::Shape{1}, {6}), false);
  const auto squeeze =
      std::make_shared<ov::op::v0::Squeeze>(data, i64_const(ov::Shape{1}, {0}));
  const auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
      param(ov::element::f32, ov::Shape{2, 3}), i64_const(ov::Shape{1}, {0}));

  const auto target = BackendTarget::from_backend(GpuBackend::OpenCL);
  const BackendCapabilities capabilities(
      target, make_opencl_operation_support_policy());
  const std::vector<std::shared_ptr<const ov::Node>> view_nodes = {
      reshape, squeeze, unsqueeze};
  for (const auto &node : view_nodes) {
    EXPECT_FALSE(resolve_opencl_catalog_source_artifact(node).has_value());
    const auto support = capabilities.query_operation({node});
    EXPECT_TRUE(support.semantic_legal);
    EXPECT_EQ(support.preferred_route_kind, LoweringRouteKind::Metadata);
    EXPECT_EQ(support.preferred_route, "metadata");
    EXPECT_EQ(support.semantic_reason, "view_only");
  }
}

TEST(GfxOpenClLayoutSourceArtifactsTest,
     ConvertHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3});
  const auto convert =
      std::make_shared<ov::op::v0::Convert>(data, ov::element::i32);

  expect_opencl_missing_kernel_unit(convert, "missing_opencl_kernel_unit");
}

TEST(GfxOpenClLayoutSourceArtifactsTest,
     MatMulHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3});
  const auto rhs = param(ov::element::f32, ov::Shape{3, 4});
  const auto matmul =
      std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);

  expect_opencl_missing_kernel_unit(matmul,
                                    "missing_opencl_matmul_kernel_unit");
}

TEST(GfxOpenClLayoutSourceArtifactsTest,
     TransposeHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto transpose = std::make_shared<ov::op::v1::Transpose>(
      data, i64_const(ov::Shape{3}, {1, 2, 0}));

  expect_opencl_missing_kernel_unit(transpose,
                                    "missing_opencl_transpose_kernel_unit");
}

TEST(GfxOpenClLayoutSourceArtifactsTest,
     SliceHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto slice = std::make_shared<ov::op::v8::Slice>(
      data, i64_const(ov::Shape{3}, {0, 1, 0}),
      i64_const(ov::Shape{3}, {2, 3, 4}), i64_const(ov::Shape{3}, {1, 1, 2}),
      i64_const(ov::Shape{3}, {0, 1, 2}));

  expect_opencl_missing_kernel_unit(slice, "missing_opencl_kernel_unit");
}

TEST(GfxOpenClLayoutSourceArtifactsTest,
     GeneratedPayloadResolverRejectsUnregisteredSourceArtifact) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3});
  const auto convert =
      std::make_shared<ov::op::v0::Convert>(data, ov::element::i32);

  compiler::KernelArtifactDescriptor descriptor;
  descriptor.kernel.backend_domain = "opencl";
  descriptor.kernel.kernel_id = "opencl/baseline/convert_f32_to_i32";
  descriptor.kernel.origin = compiler::KernelArtifactOrigin::Generated;
  descriptor.payload_kind = compiler::KernelArtifactPayloadKind::OpenClSource;

  compiler::PlannedOperation planned_convert;
  planned_convert.source_node = convert;
  planned_convert.type_name = "Convert";

  const auto resolver =
      compiler::make_opencl_kernel_artifact_payload_resolver();
  EXPECT_FALSE(static_cast<bool>(resolver(descriptor, planned_convert)));
  EXPECT_TRUE(descriptor.entry_point.empty());
}
