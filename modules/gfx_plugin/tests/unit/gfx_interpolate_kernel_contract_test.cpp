// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_opencl_source_artifact_verifier.hpp"
#include "unit/gfx_opencl_source_artifacts_contract.hpp"

#include "common/interpolate_contract.hpp"

namespace {

using InterpolateBase = ov::op::util::InterpolateBase;

std::vector<GfxOpenClSourceScalarArg> interpolate_scalar_args() {
  return {GfxOpenClSourceScalarArg::ElementCount,
          GfxOpenClSourceScalarArg::StaticU32,
          GfxOpenClSourceScalarArg::StaticU32,
          GfxOpenClSourceScalarArg::StaticU32,
          GfxOpenClSourceScalarArg::StaticU32,
          GfxOpenClSourceScalarArg::Input0Dim0,
          GfxOpenClSourceScalarArg::Input0Dim1,
          GfxOpenClSourceScalarArg::Input0Dim2,
          GfxOpenClSourceScalarArg::Input0Dim3,
          GfxOpenClSourceScalarArg::Output0Dim2,
          GfxOpenClSourceScalarArg::Output0Dim3};
}

std::shared_ptr<ov::op::v0::Interpolate>
make_v0_interpolate(ov::element::Type element_type, std::string mode) {
  const auto data = param(element_type, ov::Shape{1, 4, 16, 16});
  const auto output_shape = i64_const(ov::Shape{2}, {32, 32});
  ov::op::v0::Interpolate::Attributes attrs;
  attrs.axes = ov::AxisSet{2, 3};
  attrs.mode = std::move(mode);
  attrs.align_corners = false;
  return std::make_shared<ov::op::v0::Interpolate>(data, output_shape, attrs);
}

std::shared_ptr<ov::op::v4::Interpolate>
make_v4_interpolate(InterpolateBase::InterpolateMode mode,
                    InterpolateBase::CoordinateTransformMode coordinate_mode,
                    InterpolateBase::NearestMode nearest_mode) {
  const auto data = param(ov::element::f32, ov::Shape{1, 4, 16, 16});
  const auto output_shape = i64_const(ov::Shape{2}, {32, 32});
  const auto scales = f32_const(ov::Shape{2}, {2.f, 2.f});
  const auto axes = i64_const(ov::Shape{2}, {2, 3});

  ov::op::v4::Interpolate::InterpolateAttrs attrs;
  attrs.mode = mode;
  attrs.shape_calculation_mode = InterpolateBase::ShapeCalcMode::SIZES;
  attrs.coordinate_transformation_mode = coordinate_mode;
  attrs.nearest_mode = nearest_mode;
  return std::make_shared<ov::op::v4::Interpolate>(data, output_shape, scales,
                                                   axes, attrs);
}

std::shared_ptr<ov::op::v4::Interpolate>
make_v4_nearest_half_pixel_interpolate() {
  return make_v4_interpolate(InterpolateBase::InterpolateMode::NEAREST,
                             InterpolateBase::CoordinateTransformMode::HALF_PIXEL,
                             InterpolateBase::NearestMode::CEIL);
}

} // namespace

TEST(GfxInterpolateKernelContractTest,
     V0LinearF32UsesOpenClGeneratedSourceArtifact) {
  const auto interpolate = make_v0_interpolate(ov::element::f32, "linear");

  ov::gfx_plugin::test::OpenClSourceArtifactVerifier(interpolate)
      .expect_artifact(GfxKernelStageFamily::Resize,
                       "opencl/generated/interpolate_f32",
                       "gfx_opencl_generated_interpolate_f32", 13u, 1u,
                       interpolate_scalar_args(), {0}, {0, 0, 1, 0})
      .has_op(GfxOpenClArtifactOp::Interpolate)
      .has_input_mode(GfxOpenClArtifactInputMode::Direct)
      .supports_opencl_compiler();
}

TEST(GfxInterpolateKernelContractTest,
     V0NearestF16UsesOpenClGeneratedSourceArtifact) {
  const auto interpolate = make_v0_interpolate(ov::element::f16, "nearest");

  ov::gfx_plugin::test::OpenClSourceArtifactVerifier(interpolate)
      .expect_artifact(GfxKernelStageFamily::Resize,
                       "opencl/generated/interpolate_f16",
                       "gfx_opencl_generated_interpolate_f16", 13u, 1u,
                       interpolate_scalar_args(), {0}, {1, 0, 1, 0})
      .has_op(GfxOpenClArtifactOp::Interpolate)
      .has_input_mode(GfxOpenClArtifactInputMode::Direct)
      .supports_opencl_compiler();
}

TEST(GfxInterpolateKernelContractTest,
     V4NearestModeUsesDescriptorOwnedOpenClScalars) {
  const auto interpolate = make_v4_nearest_half_pixel_interpolate();

  ov::gfx_plugin::test::OpenClSourceArtifactVerifier(interpolate)
      .expect_artifact(GfxKernelStageFamily::Resize,
                       "opencl/generated/interpolate_f32",
                       "gfx_opencl_generated_interpolate_f32", 13u, 1u,
                       interpolate_scalar_args(), {0}, {1, 0, 1, 2})
      .has_op(GfxOpenClArtifactOp::Interpolate)
      .has_input_mode(GfxOpenClArtifactInputMode::Direct)
      .supports_opencl_compiler();
}

TEST(GfxInterpolateKernelContractTest,
     V4AlignCornersUsesSharedSemanticContract) {
  const auto interpolate =
      make_v4_interpolate(InterpolateBase::InterpolateMode::NEAREST,
                          InterpolateBase::CoordinateTransformMode::ALIGN_CORNERS,
                          InterpolateBase::NearestMode::FLOOR);
  const auto semantic =
      ov::gfx_plugin::make_interpolate_semantic_contract(interpolate);

  ASSERT_TRUE(semantic);
  EXPECT_TRUE(semantic->nearest);
  EXPECT_TRUE(semantic->align_corners);
  EXPECT_FALSE(semantic->use_half_pixel);
  EXPECT_EQ(1u, semantic->nearest_mode);
  ov::gfx_plugin::test::OpenClSourceArtifactVerifier(interpolate)
      .expect_artifact(GfxKernelStageFamily::Resize,
                       "opencl/generated/interpolate_f32",
                       "gfx_opencl_generated_interpolate_f32", 13u, 1u,
                       interpolate_scalar_args(), {0}, {1, 1, 0, 1})
      .has_op(GfxOpenClArtifactOp::Interpolate)
      .has_input_mode(GfxOpenClArtifactInputMode::Direct)
      .supports_opencl_compiler();
}

TEST(GfxInterpolateKernelContractTest,
     UnsupportedInterpolateLayoutHasNoOpenClSourceArtifact) {
  const auto data = param(ov::element::f32, ov::Shape{1, 4, 16, 16});
  const auto output_shape = i64_const(ov::Shape{2}, {1, 8});
  const auto scales = f32_const(ov::Shape{2}, {1.f, 1.f});
  const auto axes = i64_const(ov::Shape{2}, {0, 1});

  using Base = ov::op::util::InterpolateBase;
  ov::op::v4::Interpolate::InterpolateAttrs attrs;
  attrs.mode = Base::InterpolateMode::LINEAR;
  attrs.shape_calculation_mode = Base::ShapeCalcMode::SIZES;
  attrs.coordinate_transformation_mode =
      Base::CoordinateTransformMode::HALF_PIXEL;
  const auto interpolate = std::make_shared<ov::op::v4::Interpolate>(
      data, output_shape, scales, axes, attrs);

  expect_opencl_missing_kernel_unit(interpolate,
                                    "missing_opencl_interpolate_kernel_unit");
}
