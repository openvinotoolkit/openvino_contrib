// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_opencl_source_artifacts_contract.hpp"

TEST(GfxOpenClConcatSplitSourceArtifactsTest,
     ConcatHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto lhs = param(ov::element::f32, ov::Shape{1, 2, 3});
  const auto rhs = param(ov::element::f32, ov::Shape{1, 4, 3});
  const auto concat =
      std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lhs, rhs}, 1);

  expect_opencl_missing_kernel_unit(concat,
                                    "missing_opencl_concat_kernel_unit");
}

TEST(GfxOpenClConcatSplitSourceArtifactsTest,
     DynamicF16ConcatHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto lhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  const auto rhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  const auto concat =
      std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lhs, rhs}, 1);

  expect_opencl_missing_kernel_unit(concat,
                                    "missing_opencl_concat_kernel_unit");
}

TEST(GfxOpenClConcatSplitSourceArtifactsTest,
     SplitHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto data = param(ov::element::f32, ov::Shape{1, 6, 2});
  const auto split =
      std::make_shared<ov::op::v1::Split>(data, i64_const(ov::Shape{}, {1}), 3);

  expect_opencl_missing_kernel_unit(split, "missing_opencl_split_kernel_unit");
}

TEST(GfxOpenClConcatSplitSourceArtifactsTest,
     VariadicSplitHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto data = param(ov::element::f16, ov::Shape{1, 7, 2});
  const auto split = std::make_shared<ov::op::v1::VariadicSplit>(
      data, i64_const(ov::Shape{}, {1}), i64_const(ov::Shape{3}, {2, 3, 2}));

  expect_opencl_missing_kernel_unit(split, "missing_opencl_split_kernel_unit");
}
