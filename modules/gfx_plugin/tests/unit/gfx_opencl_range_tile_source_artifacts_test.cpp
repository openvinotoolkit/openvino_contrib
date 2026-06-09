// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_opencl_source_artifacts_contract.hpp"

TEST(GfxOpenClRangeTileSourceArtifactsTest,
     RangeF32ArtifactsUseDirectScalarInputsAndElementCountOnly) {
  const auto range = std::make_shared<ov::op::v4::Range>(
      f32_const(ov::Shape{}, {1.5f}), f32_const(ov::Shape{}, {6.5f}),
      f32_const(ov::Shape{}, {1.0f}), ov::element::f32);

  expect_opencl_artifact(range, GfxKernelStageFamily::GatherScatter,
                         "opencl/generated/range_f32",
                         "gfx_opencl_generated_range_f32",
                         /*arg_count=*/5,
                         /*direct_input_count=*/3,
                         {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2});
  expect_opencl_compiler_supports_generated_unit(range,
                                                 "opencl/generated/range_f32");
  expect_opencl_range_compiler_contract(range, "opencl/generated/range_f32",
                                        "gfx_opencl_generated_range_f32",
                                        /*expected_abi_arg_count=*/5);
  EXPECT_FALSE(
      make_opencl_range_source_artifact(range, "opencl/generated/range_i64")
          .has_value());
  expect_opencl_compiler_support_matches_kernel_registry(range);
}

TEST(GfxOpenClRangeTileSourceArtifactsTest,
     RangePayloadResolverRejectsMismatchedKernelUnitWithoutSourceFallback) {
  const auto range = std::make_shared<ov::op::v4::Range>(
      f32_const(ov::Shape{}, {1.5f}), f32_const(ov::Shape{}, {6.5f}),
      f32_const(ov::Shape{}, {1.0f}), ov::element::f32);

  compiler::KernelArtifactDescriptor descriptor;
  descriptor.kernel.backend_domain = "opencl";
  descriptor.kernel.kernel_id = "opencl/generated/range_i64";
  descriptor.payload_kind = compiler::KernelArtifactPayloadKind::OpenClSource;

  compiler::PlannedOperation planned_range;
  planned_range.source_node = range;
  planned_range.type_name = "Range";

  const auto resolver =
      compiler::make_opencl_kernel_artifact_payload_resolver();
  EXPECT_FALSE(static_cast<bool>(resolver(descriptor, planned_range)));
  EXPECT_TRUE(descriptor.entry_point.empty());
}

TEST(GfxOpenClRangeTileSourceArtifactsTest,
     GeneratedPayloadResolverRejectsMismatchedKernelUnitWithoutSourceFallback) {
  const auto lhs = param(ov::element::f32, ov::Shape{2, 3});
  const auto rhs = param(ov::element::f32, ov::Shape{3, 4});
  const auto matmul =
      std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);

  compiler::KernelArtifactDescriptor descriptor;
  descriptor.kernel.backend_domain = "opencl";
  descriptor.kernel.kernel_id = "opencl/generated/shapeof_i32";
  descriptor.kernel.origin = compiler::KernelArtifactOrigin::Generated;
  descriptor.payload_kind = compiler::KernelArtifactPayloadKind::OpenClSource;

  compiler::PlannedOperation planned_matmul;
  planned_matmul.source_node = matmul;
  planned_matmul.type_name = "MatMul";

  const auto resolver =
      compiler::make_opencl_kernel_artifact_payload_resolver();
  EXPECT_FALSE(static_cast<bool>(resolver(descriptor, planned_matmul)));
  EXPECT_TRUE(descriptor.entry_point.empty());
}


TEST(GfxOpenClRangeTileSourceArtifactsTest,
     RangeF16ArtifactsUsePackedF16KernelWithDirectScalarInputs) {
  const auto range = std::make_shared<ov::op::v4::Range>(
      f16_const(ov::Shape{}, {1.0f}), f16_const(ov::Shape{}, {4.0f}),
      f16_const(ov::Shape{}, {0.5f}), ov::element::f16);

  expect_opencl_artifact(range, GfxKernelStageFamily::GatherScatter,
                         "opencl/generated/range_f16",
                         "gfx_opencl_generated_range_f16",
                         /*arg_count=*/5,
                         /*direct_input_count=*/3,
                         {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2});
  expect_opencl_compiler_supports_generated_unit(range,
                                                 "opencl/generated/range_f16");
  expect_opencl_range_compiler_contract(range, "opencl/generated/range_f16",
                                        "gfx_opencl_generated_range_f16",
                                        /*expected_abi_arg_count=*/5);
  expect_opencl_compiler_support_matches_kernel_registry(range);
}

TEST(GfxOpenClRangeTileSourceArtifactsTest,
     RangeI64ArtifactsUseTheSameManifestAbiWithI64OutputKernel) {
  const auto range = std::make_shared<ov::op::v4::Range>(
      i64_const(ov::Shape{}, {0}), i64_const(ov::Shape{}, {10}),
      i64_const(ov::Shape{}, {2}), ov::element::i64);

  expect_opencl_artifact(range, GfxKernelStageFamily::GatherScatter,
                         "opencl/generated/range_i64",
                         "gfx_opencl_generated_range_i64",
                         /*arg_count=*/5,
                         /*direct_input_count=*/3,
                         {GfxOpenClSourceScalarArg::ElementCount}, {0, 1, 2});
  expect_opencl_compiler_supports_generated_unit(range,
                                                 "opencl/generated/range_i64");
  expect_opencl_range_compiler_contract(range, "opencl/generated/range_i64",
                                        "gfx_opencl_generated_range_i64",
                                        /*expected_abi_arg_count=*/5);
  expect_opencl_compiler_support_matches_kernel_registry(range);
}

TEST(GfxOpenClRangeTileSourceArtifactsTest,
     TileF32ArtifactsCarryStaticShapeAndStrideMetadata) {
  const auto data = param(ov::element::f32, ov::Shape{2, 1, 3});
  const auto tile = std::make_shared<ov::op::v0::Tile>(
      data, i64_const(ov::Shape{3}, {1, 4, 2}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 17,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      2,  4, 6, 1, // output dims padded to rank 4
      2,  1, 3, 1, // input dims padded to rank 4
      24, 6, 1, 1, // output strides padded to rank 4
      3,  3, 1, 1, // input strides padded to rank 4
  };

  expect_opencl_artifact(
      tile, GfxKernelStageFamily::Layout, "opencl/generated/tile_f32",
      "gfx_opencl_generated_tile_f32",
      /*arg_count=*/20,
      /*direct_input_count=*/1, scalar_args, {0}, static_u32_scalars);
  expect_opencl_source_excludes(
      tile, {"long", "__global long*", "gfx_opencl_generated_range_i64"});
  expect_opencl_compiler_support_matches_kernel_registry(tile);
  expect_opencl_generated_compiler_contract(
      tile, "Tile", "Tile", "opencl/generated/tile_f32",
      "gfx_opencl_generated_tile_f32",
      /*expected_abi_arg_count=*/20, {data});
}

TEST(GfxOpenClRangeTileSourceArtifactsTest,
     TileF16ArtifactsReuseStaticShapeAndStrideMetadata) {
  const auto data = param(ov::element::f16, ov::Shape{2, 1, 3});
  const auto tile = std::make_shared<ov::op::v0::Tile>(
      data, i64_const(ov::Shape{3}, {1, 4, 2}));
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 17,
                     GfxOpenClSourceScalarArg::StaticU32);
  const std::vector<uint32_t> static_u32_scalars = {
      3,           // rank
      2,  4, 6, 1, // output dims padded to rank 4
      2,  1, 3, 1, // input dims padded to rank 4
      24, 6, 1, 1, // output strides padded to rank 4
      3,  3, 1, 1, // input strides padded to rank 4
  };

  expect_opencl_artifact(
      tile, GfxKernelStageFamily::Layout, "opencl/generated/tile_f16",
      "gfx_opencl_generated_tile_f16",
      /*arg_count=*/20,
      /*direct_input_count=*/1, scalar_args, {0}, static_u32_scalars);
  expect_opencl_compiler_support_matches_kernel_registry(tile);
  expect_opencl_generated_compiler_contract(
      tile, "Tile", "Tile", "opencl/generated/tile_f16",
      "gfx_opencl_generated_tile_f16",
      /*expected_abi_arg_count=*/20, {data});
}

TEST(GfxOpenClRangeTileSourceArtifactsTest,
     TilePayloadResolverRejectsMismatchedKernelUnitWithoutSourceFallback) {
  const auto data = param(ov::element::f32, ov::Shape{2, 1, 3});
  const auto tile = std::make_shared<ov::op::v0::Tile>(
      data, i64_const(ov::Shape{3}, {1, 4, 2}));

  compiler::KernelArtifactDescriptor descriptor;
  descriptor.kernel.backend_domain = "opencl";
  descriptor.kernel.kernel_id = "opencl/generated/tile_f16";
  descriptor.kernel.origin = compiler::KernelArtifactOrigin::Generated;
  descriptor.payload_kind = compiler::KernelArtifactPayloadKind::OpenClSource;

  compiler::PlannedOperation planned_tile;
  planned_tile.source_node = tile;
  planned_tile.type_name = "Tile";

  const auto resolver =
      compiler::make_opencl_kernel_artifact_payload_resolver();
  EXPECT_FALSE(static_cast<bool>(resolver(descriptor, planned_tile)));
  EXPECT_TRUE(descriptor.entry_point.empty());
}

TEST(GfxOpenClRangeTileSourceArtifactsTest,
     DynamicF16TileUsesRuntimeInputAndOutputDimsUnderSourceManifest) {
  const auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 3});
  const auto repeats = std::make_shared<ov::op::v0::Parameter>(
      ov::element::i64, ov::PartialShape{3});
  const auto tile = std::make_shared<ov::op::v0::Tile>(data, repeats);

  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::Output0Dim0,
      GfxOpenClSourceScalarArg::Output0Dim1,
      GfxOpenClSourceScalarArg::Output0Dim2,
      GfxOpenClSourceScalarArg::Output0Dim3,
      GfxOpenClSourceScalarArg::Input0Dim0,
      GfxOpenClSourceScalarArg::Input0Dim1,
      GfxOpenClSourceScalarArg::Input0Dim2,
      GfxOpenClSourceScalarArg::Input0Dim3};

  expect_opencl_artifact(tile, GfxKernelStageFamily::Layout,
                         "opencl/generated/tile_dynamic_f16",
                         "gfx_opencl_generated_tile_dynamic_f16",
                         /*arg_count=*/12,
                         /*direct_input_count=*/1, scalar_args, {0}, {3});
  expect_opencl_compiler_support_matches_kernel_registry(tile);
}

TEST(GfxOpenClRangeTileSourceArtifactsTest,
     DynamicF32TileUsesRuntimeInputAndOutputDimsUnderSourceManifest) {
  const auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f32, ov::PartialShape{1, -1, 3});
  const auto repeats = std::make_shared<ov::op::v0::Parameter>(
      ov::element::i64, ov::PartialShape{3});
  const auto tile = std::make_shared<ov::op::v0::Tile>(data, repeats);

  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount,
      GfxOpenClSourceScalarArg::StaticU32,
      GfxOpenClSourceScalarArg::Output0Dim0,
      GfxOpenClSourceScalarArg::Output0Dim1,
      GfxOpenClSourceScalarArg::Output0Dim2,
      GfxOpenClSourceScalarArg::Output0Dim3,
      GfxOpenClSourceScalarArg::Input0Dim0,
      GfxOpenClSourceScalarArg::Input0Dim1,
      GfxOpenClSourceScalarArg::Input0Dim2,
      GfxOpenClSourceScalarArg::Input0Dim3};

  expect_opencl_artifact(tile, GfxKernelStageFamily::Layout,
                         "opencl/generated/tile_dynamic_f32",
                         "gfx_opencl_generated_tile_dynamic_f32",
                         /*arg_count=*/12,
                         /*direct_input_count=*/1, scalar_args, {0}, {3});
  expect_opencl_source_excludes(
      tile, {"long", "__global long*", "gfx_opencl_generated_range_i64"});
  expect_opencl_compiler_support_matches_kernel_registry(tile);
}


TEST(GfxOpenClRangeTileSourceArtifactsTest,
     DynamicF16SliceHasNoOpenClSourceArtifactWithoutKernelUnit) {
  auto data = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::PartialShape{1, -1, 4});
  auto starts = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                        ov::PartialShape{3});
  auto ends = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                      ov::PartialShape{3});
  auto steps = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                       ov::PartialShape{3});
  const auto slice =
      std::make_shared<ov::op::v8::Slice>(data, starts, ends, steps);

  expect_opencl_missing_kernel_unit(slice, "missing_opencl_kernel_unit");
}

TEST(GfxOpenClRangeTileSourceArtifactsTest,
     DynamicI64UnitRangeUsesManifestAbiAndRegisteredKernelUnit) {
  auto stop = std::make_shared<ov::op::v0::Parameter>(ov::element::i64,
                                                      ov::PartialShape{});
  const auto start = i64_const(ov::Shape{}, {0});
  const auto step = i64_const(ov::Shape{}, {1});
  const auto range =
      std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::i64);

  expect_opencl_artifact(range, GfxKernelStageFamily::GatherScatter,
                         "opencl/generated/range_i64_unit_dynamic",
                         "gfx_opencl_generated_range_i64_unit",
                         /*arg_count=*/3,
                         /*direct_input_count=*/1,
                         {GfxOpenClSourceScalarArg::ElementCount}, {1});
  expect_opencl_compiler_supports_generated_unit(
      range, "opencl/generated/range_i64_unit_dynamic");
  expect_opencl_range_compiler_contract(
      range, "opencl/generated/range_i64_unit_dynamic",
      "gfx_opencl_generated_range_i64_unit",
      /*expected_abi_arg_count=*/3, ov::ParameterVector{stop});
  expect_opencl_compiler_support_matches_kernel_registry(range);
}
