// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_opencl_source_artifacts_contract.hpp"

TEST(GfxOpenClGatherScatterSourceArtifactsTest,
     GatherHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i64, ov::Shape{2});
  const auto gather = std::make_shared<ov::op::v8::Gather>(
      data, indices, i64_const(ov::Shape{}, {1}));

  expect_opencl_missing_kernel_unit(gather, "missing_opencl_kernel_unit");
}

TEST(GfxOpenClGatherScatterSourceArtifactsTest,
     GatherElementsHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i32, ov::Shape{2, 3, 2});
  const auto gather =
      std::make_shared<ov::op::v6::GatherElements>(data, indices, -1);

  expect_opencl_missing_kernel_unit(gather, "missing_opencl_kernel_unit");
}

TEST(GfxOpenClGatherScatterSourceArtifactsTest,
     GatherNDHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i32, ov::Shape{2, 1});
  const auto gather = std::make_shared<ov::op::v8::GatherND>(data, indices);

  expect_opencl_missing_kernel_unit(gather, "missing_opencl_kernel_unit");
}

TEST(GfxOpenClGatherScatterSourceArtifactsTest,
     ScatterUpdateHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i64, ov::Shape{2});
  const auto updates = param(ov::element::f32, ov::Shape{2, 2, 4});
  const auto scatter = std::make_shared<ov::op::v3::ScatterUpdate>(
      data, indices, updates, i64_const(ov::Shape{}, {1}));

  expect_opencl_missing_kernel_unit(scatter, "missing_opencl_kernel_unit");
}

TEST(GfxOpenClGatherScatterSourceArtifactsTest,
     ScatterElementsHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i32, ov::Shape{2, 2, 4});
  const auto updates = param(ov::element::f32, ov::Shape{2, 2, 4});
  const auto scatter = std::make_shared<ov::op::v3::ScatterElementsUpdate>(
      data, indices, updates, i64_const(ov::Shape{}, {1}));

  expect_opencl_missing_kernel_unit(scatter, "missing_opencl_kernel_unit");
}

TEST(GfxOpenClGatherScatterSourceArtifactsTest,
     ScatterNDHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto data = param(ov::element::f32, ov::Shape{2, 3, 4});
  const auto indices = param(ov::element::i64, ov::Shape{2, 2});
  const auto updates = param(ov::element::f32, ov::Shape{2, 4});
  const auto scatter =
      std::make_shared<ov::op::v3::ScatterNDUpdate>(data, indices, updates);

  expect_opencl_missing_kernel_unit(scatter, "missing_opencl_kernel_unit");
}

TEST(GfxOpenClGatherScatterSourceArtifactsTest,
     DynamicBroadcastHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  const auto target_shape = std::make_shared<ov::op::v0::Parameter>(
      ov::element::i64, ov::PartialShape{3});
  const auto broadcast = std::make_shared<ov::op::v3::Broadcast>(
      data, target_shape, ov::op::BroadcastType::BIDIRECTIONAL);

  expect_opencl_missing_kernel_unit(broadcast, "missing_opencl_kernel_unit");
}

TEST(GfxOpenClGatherScatterSourceArtifactsTest,
     DynamicStridedSliceHasNoOpenClSourceArtifactWithoutKernelUnit) {
  const auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  const auto end = std::make_shared<ov::op::v0::Parameter>(
      ov::element::i64, ov::PartialShape{3});
  const auto begin = i64_const(ov::Shape{3}, {0, 0, 0});
  const auto strides = i64_const(ov::Shape{3}, {1, 1, 1});
  const auto slice = std::make_shared<ov::op::v1::StridedSlice>(
      data, begin, end, strides, std::vector<int64_t>{0, 0, 0},
      std::vector<int64_t>{0, 0, 0});

  expect_opencl_missing_kernel_unit(slice, "missing_opencl_kernel_unit");
}
