// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include "runtime/gfx_mpsrt_model.hpp"
#include "mlir/gfx_apple_vendor_descriptors.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/gfx_mpsrt_runtime_abi_pipeline.hpp"
#include "mlir/msl_codegen.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "runtime/gfx_mpsrt_abi.hpp"
#include "runtime/gfx_mpsrt_builder_plan.hpp"
#include "runtime/gfx_mpsrt_kernel_manifest_adapter.hpp"
#include "runtime/gfx_mpsrt_plan.hpp"
#include "runtime/gfx_mpsrt_program.hpp"
#include "runtime/gfx_mpsrt_storage_bridge.hpp"
#include "runtime/gfx_stage_policy.hpp"

using namespace ov::gfx_plugin;

namespace runtime_mpsrt = ov::gfx_plugin::mpsrt;

namespace {

class FakeDeviceInfoBufferManager final : public GpuBufferManager {
public:
  explicit FakeDeviceInfoBufferManager(GpuExecutionDeviceInfo info)
      : m_info(std::move(info)) {}

  std::optional<GpuExecutionDeviceInfo>
  query_execution_device_info() const override {
    return m_info;
  }

private:
  GpuExecutionDeviceInfo m_info;
};

std::shared_ptr<const ov::Node> make_pointwise_conv_node() {
  auto input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::Shape{1, 64, 64, 64});
  auto weights =
      ov::op::v0::Constant::create(ov::element::f16, ov::Shape{128, 64, 1, 1},
                                   std::vector<float>(128 * 64, 1.f));
  return std::make_shared<ov::op::v1::Convolution>(
      input, weights, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0},
      ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
}

std::shared_ptr<const ov::Node> make_non_aligned_channel_conv_node() {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                       ov::Shape{1, 3, 32, 32});
  auto weights =
      ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2, 3, 3, 3},
                                   std::vector<float>(2 * 3 * 3 * 3, 1.f));
  return std::make_shared<ov::op::v1::Convolution>(
      input, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
}

std::shared_ptr<const ov::Node> make_dilated_non_aligned_channel_conv_node() {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                       ov::Shape{1, 3, 32, 32});
  auto weights =
      ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2, 3, 3, 3},
                                   std::vector<float>(2 * 3 * 3 * 3, 1.f));
  return std::make_shared<ov::op::v1::Convolution>(
      input, weights, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0},
      ov::CoordinateDiff{0, 0}, ov::Strides{3, 1});
}

std::shared_ptr<const ov::Node> make_depthwise_group_conv_node() {
  auto input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::Shape{1, 32, 32, 32});
  auto weights = ov::op::v0::Constant::create(
      ov::element::f16, ov::Shape{32, 1, 1, 1, 1}, std::vector<float>(32, 1.f));
  return std::make_shared<ov::op::v1::GroupConvolution>(
      input, weights, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0},
      ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
}

std::shared_ptr<const ov::Node> make_large_add_node() {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  return std::make_shared<ov::op::v1::Add>(lhs, rhs);
}

std::shared_ptr<const ov::Node> make_matmul_node() {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 128, 256});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 256, 64});
  return std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);
}

std::shared_ptr<const ov::Node> make_aligned_maxpool_node() {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                       ov::Shape{1, 4, 16, 16});
  return std::make_shared<ov::op::v1::MaxPool>(
      input, ov::Strides{2, 2}, ov::Shape{0, 0}, ov::Shape{0, 0},
      ov::Shape{2, 2}, ov::op::RoundingType::FLOOR);
}

std::shared_ptr<const ov::Node> make_bilinear_interpolate_node() {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                       ov::Shape{1, 4, 16, 16});
  auto output_shape = ov::op::v0::Constant::create(
      ov::element::i64, ov::Shape{2}, std::vector<int64_t>{32, 32});
  ov::op::v0::Interpolate::Attributes attrs;
  attrs.axes = ov::AxisSet{2, 3};
  attrs.mode = "linear";
  attrs.align_corners = false;
  return std::make_shared<ov::op::v0::Interpolate>(input, output_shape, attrs);
}

std::shared_ptr<const ov::Node>
make_v4_non_spatial_bilinear_interpolate_node() {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                       ov::Shape{1, 4, 16, 16});
  auto output_shape = ov::op::v0::Constant::create(
      ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 4});
  auto scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2},
                                             std::vector<float>{1.f, 1.f});
  auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2},
                                           std::vector<int64_t>{0, 1});
  using Base = ov::op::util::InterpolateBase;
  ov::op::v4::Interpolate::InterpolateAttrs attrs;
  attrs.mode = Base::InterpolateMode::LINEAR;
  attrs.shape_calculation_mode = Base::ShapeCalcMode::SIZES;
  attrs.coordinate_transformation_mode =
      Base::CoordinateTransformMode::HALF_PIXEL;
  return std::make_shared<ov::op::v4::Interpolate>(input, output_shape, scales,
                                                   axes, attrs);
}

std::shared_ptr<const ov::Node> make_last_dim_softmax_node() {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                       ov::Shape{2, 8, 16});
  return std::make_shared<ov::op::v1::Softmax>(input, 2);
}

std::shared_ptr<const ov::Node> make_last_dim_topk_node() {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                       ov::Shape{2, 8, 16});
  auto k = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{},
                                        std::vector<int32_t>{4});
  return std::make_shared<ov::op::v11::TopK>(input, k, 2, ov::op::TopKMode::MAX,
                                             ov::op::TopKSortType::SORT_VALUES,
                                             ov::element::i32);
}

std::shared_ptr<const ov::Node> make_large_concat_node() {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 64, 80, 80});
  return std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lhs, rhs}, 1);
}

std::shared_ptr<const ov::Node> make_large_transpose_node() {
  auto input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::Shape{1, 32, 80, 80});
  auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4},
                                           std::vector<int64_t>{0, 2, 3, 1});
  return std::make_shared<ov::op::v1::Transpose>(input, perm);
}

std::shared_ptr<const ov::Node> make_large_chunked_conv_node() {
  auto input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::Shape{1, 64, 64, 64});
  auto weights =
      ov::op::v0::Constant::create(ov::element::f16, ov::Shape{128, 64, 5, 5},
                                   std::vector<float>(128 * 64 * 5 * 5, 1.f));
  return std::make_shared<ov::op::v1::Convolution>(
      input, weights, ov::Strides{1, 1}, ov::CoordinateDiff{2, 2},
      ov::CoordinateDiff{2, 2}, ov::Strides{1, 1});
}

std::shared_ptr<const ov::Node> make_medium_chunked_conv_node() {
  auto input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::Shape{1, 64, 32, 32});
  auto weights =
      ov::op::v0::Constant::create(ov::element::f16, ov::Shape{64, 64, 5, 5},
                                   std::vector<float>(64 * 64 * 5 * 5, 1.f));
  return std::make_shared<ov::op::v1::Convolution>(
      input, weights, ov::Strides{1, 1}, ov::CoordinateDiff{2, 2},
      ov::CoordinateDiff{2, 2}, ov::Strides{1, 1});
}

std::shared_ptr<const ov::Node> make_light_depthwise_group_conv_node() {
  auto input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::Shape{1, 64, 32, 32});
  auto weights =
      ov::op::v0::Constant::create(ov::element::f16, ov::Shape{64, 1, 1, 3, 3},
                                   std::vector<float>(64 * 3 * 3, 1.f));
  return std::make_shared<ov::op::v1::GroupConvolution>(
      input, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
}

std::shared_ptr<const ov::Node> make_light_spatial3x3_conv_node() {
  auto input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::Shape{1, 16, 160, 160});
  auto weights =
      ov::op::v0::Constant::create(ov::element::f16, ov::Shape{8, 16, 3, 3},
                                   std::vector<float>(8 * 16 * 3 * 3, 1.f));
  return std::make_shared<ov::op::v1::Convolution>(
      input, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
}

std::pair<std::shared_ptr<const ov::Node>, std::shared_ptr<const ov::Node>>
make_light_spatial3x3_conv_chain() {
  auto input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::Shape{1, 16, 160, 160});
  auto weights0 =
      ov::op::v0::Constant::create(ov::element::f16, ov::Shape{8, 16, 3, 3},
                                   std::vector<float>(8 * 16 * 3 * 3, 1.f));
  auto conv0 = std::make_shared<ov::op::v1::Convolution>(
      input, weights0, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
  auto weights1 =
      ov::op::v0::Constant::create(ov::element::f16, ov::Shape{8, 8, 3, 3},
                                   std::vector<float>(8 * 8 * 3 * 3, 1.f));
  auto conv1 = std::make_shared<ov::op::v1::Convolution>(
      conv0, weights1, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
  return {conv0, conv1};
}

std::shared_ptr<const ov::Node> make_concat_fed_spatial3x3_conv_node() {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 8, 160, 160});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                     ov::Shape{1, 8, 160, 160});
  auto concat =
      std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lhs, rhs}, 1);
  auto weights =
      ov::op::v0::Constant::create(ov::element::f16, ov::Shape{8, 16, 3, 3},
                                   std::vector<float>(8 * 16 * 3 * 3, 1.f));
  return std::make_shared<ov::op::v1::Convolution>(
      concat, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
}

std::shared_ptr<const ov::Node> make_chunked_conv_with_reshape_consumer() {
  auto input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f16, ov::Shape{1, 64, 32, 32});
  auto weights =
      ov::op::v0::Constant::create(ov::element::f16, ov::Shape{64, 64, 5, 5},
                                   std::vector<float>(64 * 64 * 5 * 5, 1.f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      input, weights, ov::Strides{1, 1}, ov::CoordinateDiff{2, 2},
      ov::CoordinateDiff{2, 2}, ov::Strides{1, 1});
  auto pattern = ov::op::v0::Constant::create(
      ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 32, 32, 64});
  (void)std::make_shared<ov::op::v1::Reshape>(conv, pattern, false);
  return conv;
}

} // namespace

TEST(
    GfxStagePolicyTest,
    VulkanPointwiseConvolutionStaysOnSharedMlirRouteAndAllowsBiasAndReluFusion) {
  const auto conv = make_pointwise_conv_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Vulkan, "Convolution", conv, ov::element::f16,
      /*has_bias=*/true,
      /*has_activation=*/true,
      /*has_batchnorm=*/false, {});

  EXPECT_TRUE(plan.post_ops.bias);
  EXPECT_TRUE(plan.post_ops.activation);
  EXPECT_TRUE(plan.post_ops.batchnorm);
  EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::None);
  EXPECT_EQ(plan.conv.family, GfxConvFamily::Unknown);
  EXPECT_EQ(plan.conv.algorithm.kind, GfxConvAlgorithmKind::None);
  EXPECT_FALSE(plan.execution.submit.isolate);
  EXPECT_EQ(plan.execution.submit.weight, 4u);
}

TEST(GfxStagePolicyTest, MetalConvolutionPlansAppleMpsImageStorage) {
  const auto conv = make_pointwise_conv_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "Convolution", conv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});

  EXPECT_EQ(plan.archetype, GfxStageArchetype::Convolution);
  EXPECT_EQ(plan.placement.domain, GfxStageBackendDomain::AppleMps);
  EXPECT_EQ(plan.placement.storage, GfxStageStorageKind::Image);
  EXPECT_TRUE(plan.placement.uses_vendor_primitive);
  EXPECT_FALSE(plan.placement.uses_custom_kernel);
  EXPECT_TRUE(plan.post_ops.bias);
  EXPECT_TRUE(plan.post_ops.activation);
}

TEST(GfxStagePolicyTest,
     MetalConvolutionWithGenericActivationStaysMslUntilActivationKindIsKnown) {
  const auto conv = make_pointwise_conv_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "Convolution", conv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/true,
      /*has_batchnorm=*/false, {});

  EXPECT_EQ(plan.archetype, GfxStageArchetype::Convolution);
  EXPECT_EQ(plan.placement.domain, GfxStageBackendDomain::AppleMsl);
  EXPECT_EQ(plan.placement.storage, GfxStageStorageKind::Buffer);
  EXPECT_FALSE(plan.placement.uses_vendor_primitive);
  EXPECT_TRUE(plan.placement.uses_custom_kernel);
}

TEST(GfxStagePolicyTest,
     MetalConvolutionWithNonAlignedChannelsStillPlansAppleMpsImageStorage) {
  const auto conv = make_non_aligned_channel_conv_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "Convolution", conv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});

  EXPECT_EQ(plan.archetype, GfxStageArchetype::Convolution);
  EXPECT_EQ(plan.placement.domain, GfxStageBackendDomain::AppleMps);
  EXPECT_EQ(plan.placement.storage, GfxStageStorageKind::Image);
  EXPECT_TRUE(plan.placement.uses_vendor_primitive);
  EXPECT_FALSE(plan.placement.uses_custom_kernel);

  const auto desc = gfx_mpsrt_make_stage_desc(plan, "Convolution");
  EXPECT_EQ(desc.kind, GfxMpsrtStageKind::MPSConv2D);
  EXPECT_EQ(desc.layout, GfxMpsrtLayout::NHWC4);
  EXPECT_EQ(desc.specialization_key, "apple_mps:image:Convolution");
}

TEST(GfxStagePolicyTest, MetalDilatedConvolutionPlansAppleMpsImageStorage) {
  const auto conv = make_dilated_non_aligned_channel_conv_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "Convolution", conv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});

  EXPECT_EQ(plan.archetype, GfxStageArchetype::Convolution);
  EXPECT_EQ(plan.placement.domain, GfxStageBackendDomain::AppleMps);
  EXPECT_EQ(plan.placement.storage, GfxStageStorageKind::Image);
  EXPECT_TRUE(plan.placement.uses_vendor_primitive);
  EXPECT_FALSE(plan.placement.uses_custom_kernel);
}

TEST(GfxStagePolicyTest,
     MetalDepthwiseGroupConvolutionPlansAppleMpsImageStorage) {
  const auto gconv = make_depthwise_group_conv_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "GroupConvolution", gconv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});

  EXPECT_EQ(plan.archetype, GfxStageArchetype::GroupConvolution);
  EXPECT_EQ(plan.placement.domain, GfxStageBackendDomain::AppleMps);
  EXPECT_EQ(plan.placement.storage, GfxStageStorageKind::Image);
  EXPECT_TRUE(plan.placement.uses_vendor_primitive);
  EXPECT_FALSE(plan.placement.uses_custom_kernel);

  const auto desc = gfx_mpsrt_make_stage_desc(plan, "GroupConvolution");
  EXPECT_EQ(desc.kind, GfxMpsrtStageKind::MPSGroupConv2D);
  EXPECT_EQ(desc.stage_manifest.stage_family,
            GfxKernelStageFamily::GroupConvolution);
  EXPECT_EQ(desc.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMps);
  EXPECT_EQ(desc.stage_manifest.execution_kind,
            GfxKernelExecutionKind::VendorPrimitive);
}

TEST(GfxStagePolicyTest,
     MetalBilinearInterpolatePlansAppleMpsResizeImageStorage) {
  const auto resize = make_bilinear_interpolate_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "Interpolate", resize, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});

  EXPECT_EQ(plan.placement.domain, GfxStageBackendDomain::AppleMps);
  EXPECT_EQ(plan.placement.storage, GfxStageStorageKind::Image);
  EXPECT_TRUE(plan.placement.uses_vendor_primitive);
  EXPECT_FALSE(plan.placement.uses_custom_kernel);

  const auto desc = gfx_mpsrt_make_stage_desc(plan, "Interpolate");
  EXPECT_EQ(desc.kind, GfxMpsrtStageKind::MPSResize2D);
  EXPECT_EQ(desc.stage_manifest.stage_family, GfxKernelStageFamily::Resize);
  EXPECT_EQ(desc.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMps);
  EXPECT_EQ(desc.stage_manifest.execution_kind,
            GfxKernelExecutionKind::VendorPrimitive);
  EXPECT_EQ(desc.stage_manifest.storage, GfxKernelStorageKind::Image);
}

TEST(GfxStagePolicyTest,
     AppleMpsResize2DDescriptorAcceptsSpatialBilinearInterpolate) {
  GfxMpsrtResize2DAbiDesc desc{};
  ASSERT_TRUE(
      gfx_apple_make_mps_resize2d_desc(make_bilinear_interpolate_node(), desc));
  EXPECT_EQ(desc.nearest, 0u);
  EXPECT_EQ(desc.align_corners, 0u);
  EXPECT_EQ(desc.half_pixel_centers, 1u);
}

TEST(GfxStagePolicyTest,
     AppleMpsResize2DDescriptorRejectsNonSpatialInterpolateAxes) {
  GfxMpsrtResize2DAbiDesc desc{};
  EXPECT_FALSE(gfx_apple_make_mps_resize2d_desc(
      make_v4_non_spatial_bilinear_interpolate_node(), desc));
}

TEST(GfxStagePolicyTest,
     AppleMpsPoolSoftmaxTopKDescriptorsAcceptSupportedVendorCases) {
  GfxMpsrtPool2DAbiDesc pool_desc{};
  ASSERT_TRUE(
      gfx_apple_make_mps_pool2d_desc(make_aligned_maxpool_node(), pool_desc));
  EXPECT_EQ(pool_desc.is_avg, 0u);
  EXPECT_EQ(pool_desc.kernel[0], 2u);
  EXPECT_EQ(pool_desc.strides[1], 2u);

  GfxMpsrtSoftmaxAbiDesc softmax_desc{};
  ASSERT_TRUE(gfx_apple_make_mps_softmax_desc(make_last_dim_softmax_node(),
                                              softmax_desc));
  EXPECT_EQ(softmax_desc.axis, 2u);
  EXPECT_EQ(softmax_desc.log_softmax, 0u);

  GfxMpsrtTopKAbiDesc topk_desc{};
  ASSERT_TRUE(
      gfx_apple_make_mps_topk_desc(make_last_dim_topk_node(), topk_desc));
  EXPECT_EQ(topk_desc.axis, 2u);
  EXPECT_EQ(topk_desc.k, 4u);
  EXPECT_EQ(topk_desc.mode_max, 1u);
  EXPECT_EQ(topk_desc.sort_type, 1u);
}

TEST(GfxStagePolicyTest, MetalMatMulPlansAppleMpsMatrixStorage) {
  const auto matmul = make_matmul_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "MatMul", matmul, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});

  EXPECT_EQ(plan.archetype, GfxStageArchetype::MatMul);
  EXPECT_EQ(plan.placement.domain, GfxStageBackendDomain::AppleMps);
  EXPECT_EQ(plan.placement.storage, GfxStageStorageKind::Matrix);
  EXPECT_EQ(plan.placement.specialization_key, "apple_mps:matrix:MatMul");
}

TEST(GfxStagePolicyTest, MetalBinaryElementwiseStaysInMslBufferDomain) {
  const auto add = make_large_add_node();
  GfxStageRuntimeTraits traits{};
  traits.binary_chunked = true;
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "Add", add, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, traits);

  EXPECT_EQ(plan.archetype, GfxStageArchetype::BinaryElementwise);
  EXPECT_EQ(plan.placement.domain, GfxStageBackendDomain::AppleMsl);
  EXPECT_EQ(plan.placement.storage, GfxStageStorageKind::Buffer);
  EXPECT_FALSE(plan.placement.uses_vendor_primitive);
  EXPECT_TRUE(plan.placement.uses_custom_kernel);
}

TEST(GfxStagePolicyTest, VulkanPlacementRemainsSharedSpirvBufferDomain) {
  const auto add = make_large_add_node();
  GfxStageRuntimeTraits traits{};
  traits.binary_chunked = true;
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Vulkan, "Add", add, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, traits);

  EXPECT_EQ(plan.placement.domain, GfxStageBackendDomain::Spirv);
  EXPECT_EQ(plan.placement.storage, GfxStageStorageKind::Buffer);
  EXPECT_TRUE(plan.placement.uses_custom_kernel);
}

TEST(GfxStagePolicyTest,
     MpsrtImageTensorDescriptorUsesLogicalNchwShapeAndNhwc4StorageContract) {
  const auto desc = gfx_mpsrt_make_tensor_desc(
      {1, 64, 32, 16}, ov::element::f16, GfxStageStorageKind::Image);

  EXPECT_EQ(desc.dtype, GfxMpsrtDType::F16);
  EXPECT_EQ(desc.storage, GfxMpsrtStorage::Image);
  EXPECT_EQ(desc.layout, GfxMpsrtLayout::NHWC4);
  EXPECT_EQ(desc.rank, 4u);
  EXPECT_EQ(desc.dims[0], 1u);
  EXPECT_EQ(desc.dims[1], 64u);
  EXPECT_EQ(desc.dims[2], 32u);
  EXPECT_EQ(desc.dims[3], 16u);
  EXPECT_EQ(desc.strides[0], 64 * 32 * 16);
  EXPECT_EQ(desc.strides[1], 32 * 16);
  EXPECT_EQ(desc.strides[2], 16);
  EXPECT_EQ(desc.strides[3], 1);
  EXPECT_EQ(desc.byte_length, 1u * 64u * 32u * 16u * 2u);
  EXPECT_EQ(desc.image_batch, 1u);
  EXPECT_EQ(desc.image_feature_channels, 64u);
  EXPECT_EQ(desc.image_height, 32u);
  EXPECT_EQ(desc.image_width, 16u);
}

TEST(GfxStagePolicyTest,
     MpsrtImageStorageBridgeContractClassifiesExternalIoDirections) {
  const auto desc = gfx_mpsrt_to_abi_desc(gfx_mpsrt_make_tensor_desc(
      {1, 64, 32, 16}, ov::element::f16, GfxStageStorageKind::Image,
      GfxMpsrtTensorFlagExternalIo));

  EXPECT_TRUE(gfx_mpsrt_tensor_is_image(desc));
  EXPECT_TRUE(gfx_mpsrt_image_bridge_supported(desc));
  EXPECT_EQ(gfx_mpsrt_external_image_bridge_direction(false),
            GfxMpsrtStorageBridgeDirection::BufferToImage);
  EXPECT_EQ(gfx_mpsrt_external_image_bridge_direction(true),
            GfxMpsrtStorageBridgeDirection::ImageToBuffer);

  GfxMpsrtStorageBridgeDesc input_bridge{};
  ASSERT_TRUE(gfx_mpsrt_make_image_bridge_desc(
      7u, desc, gfx_mpsrt_external_image_bridge_direction(false),
      input_bridge));
  EXPECT_EQ(input_bridge.value, 7u);
  EXPECT_EQ(input_bridge.direction,
            GfxMpsrtStorageBridgeDirection::BufferToImage);
  EXPECT_EQ(input_bridge.source_storage, GfxMpsrtStorage::Buffer);
  EXPECT_EQ(input_bridge.target_storage, GfxMpsrtStorage::Image);
  EXPECT_STREQ(gfx_mpsrt_storage_bridge_direction_name(input_bridge.direction),
               "buffer_to_image");

  GfxMpsrtStorageBridgeDesc output_bridge{};
  ASSERT_TRUE(gfx_mpsrt_make_image_bridge_desc(
      8u, desc, gfx_mpsrt_external_image_bridge_direction(true),
      output_bridge));
  EXPECT_EQ(output_bridge.value, 8u);
  EXPECT_EQ(output_bridge.direction,
            GfxMpsrtStorageBridgeDirection::ImageToBuffer);
  EXPECT_EQ(output_bridge.source_storage, GfxMpsrtStorage::Image);
  EXPECT_EQ(output_bridge.target_storage, GfxMpsrtStorage::Buffer);
  EXPECT_STREQ(gfx_mpsrt_storage_bridge_direction_name(output_bridge.direction),
               "image_to_buffer");
}

TEST(GfxStagePolicyTest,
     MpsrtImageStorageBridgeRejectsNonImageOrIncompleteTensor) {
  const auto matrix_desc = gfx_mpsrt_to_abi_desc(gfx_mpsrt_make_tensor_desc(
      {2, 128, 64}, ov::element::f16, GfxStageStorageKind::Matrix));
  EXPECT_FALSE(gfx_mpsrt_tensor_is_image(matrix_desc));
  EXPECT_FALSE(gfx_mpsrt_image_bridge_supported(matrix_desc));

  auto incomplete_image = gfx_mpsrt_to_abi_desc(gfx_mpsrt_make_tensor_desc(
      {1, 64, 32, 16}, ov::element::f16, GfxStageStorageKind::Image));
  incomplete_image.image_width = 0;
  EXPECT_TRUE(gfx_mpsrt_tensor_is_image(incomplete_image));
  EXPECT_FALSE(gfx_mpsrt_image_bridge_supported(incomplete_image));

  GfxMpsrtStorageBridgeDesc bridge{};
  EXPECT_FALSE(gfx_mpsrt_make_image_bridge_desc(
      1u, matrix_desc, GfxMpsrtStorageBridgeDirection::BufferToImage, bridge));
  EXPECT_FALSE(gfx_mpsrt_make_image_bridge_desc(
      1u, incomplete_image, GfxMpsrtStorageBridgeDirection::BufferToImage,
      bridge));
  const auto valid_image = gfx_mpsrt_to_abi_desc(gfx_mpsrt_make_tensor_desc(
      {1, 64, 32, 16}, ov::element::f16, GfxStageStorageKind::Image));
  EXPECT_FALSE(gfx_mpsrt_make_image_bridge_desc(
      1u, valid_image, GfxMpsrtStorageBridgeDirection::Unknown, bridge));
}

TEST(GfxStagePolicyTest,
     MpsrtStorageBridgeContractCoversMatrixNdarrayAndAlias) {
  const auto matrix_desc = gfx_mpsrt_to_abi_desc(gfx_mpsrt_make_tensor_desc(
      {2, 128, 64}, ov::element::f16, GfxStageStorageKind::Matrix,
      GfxMpsrtTensorFlagExternalIo));
  EXPECT_TRUE(gfx_mpsrt_matrix_bridge_supported(matrix_desc));
  EXPECT_EQ(gfx_mpsrt_external_bridge_direction_for_storage(
                GfxMpsrtStorage::Matrix,
                /*external_output=*/false),
            GfxMpsrtStorageBridgeDirection::BufferToMatrix);
  EXPECT_EQ(
      gfx_mpsrt_external_bridge_direction_for_storage(GfxMpsrtStorage::Matrix,
                                                      /*external_output=*/true),
      GfxMpsrtStorageBridgeDirection::MatrixToBuffer);

  GfxMpsrtStorageBridgeDesc matrix_bridge{};
  ASSERT_TRUE(gfx_mpsrt_make_storage_bridge_desc(
      11u, matrix_desc, GfxMpsrtStorageBridgeDirection::BufferToMatrix,
      matrix_bridge));
  EXPECT_EQ(matrix_bridge.source_storage, GfxMpsrtStorage::Buffer);
  EXPECT_EQ(matrix_bridge.target_storage, GfxMpsrtStorage::Matrix);
  EXPECT_STREQ(gfx_mpsrt_storage_bridge_direction_name(matrix_bridge.direction),
               "buffer_to_matrix");
  EXPECT_EQ(gfx_mpsrt_storage_bridge_direction_from_name("matrix_to_buffer"),
            GfxMpsrtStorageBridgeDirection::MatrixToBuffer);
  EXPECT_FALSE(gfx_mpsrt_make_image_bridge_desc(
      11u, matrix_desc, GfxMpsrtStorageBridgeDirection::BufferToMatrix,
      matrix_bridge));

  const auto ndarray_desc = gfx_mpsrt_to_abi_desc(gfx_mpsrt_make_tensor_desc(
      {2, 4, 8}, ov::element::f32, GfxStageStorageKind::NDArray,
      GfxMpsrtTensorFlagExternalIo));
  ASSERT_TRUE(gfx_mpsrt_make_storage_bridge_desc(
      12u, ndarray_desc, GfxMpsrtStorageBridgeDirection::BufferToNDArray,
      matrix_bridge));
  EXPECT_EQ(matrix_bridge.target_storage, GfxMpsrtStorage::NDArray);
  EXPECT_STREQ(gfx_mpsrt_storage_bridge_direction_name(matrix_bridge.direction),
               "buffer_to_ndarray");

  auto alias_desc = gfx_mpsrt_to_abi_desc(gfx_mpsrt_make_tensor_desc(
      {2, 4}, ov::element::f16, GfxStageStorageKind::Alias));
  alias_desc.alias_of = 3u;
  ASSERT_TRUE(gfx_mpsrt_make_storage_bridge_desc(
      13u, alias_desc, GfxMpsrtStorageBridgeDirection::Alias, matrix_bridge));
  EXPECT_EQ(matrix_bridge.source_storage, GfxMpsrtStorage::Alias);
  EXPECT_EQ(matrix_bridge.target_storage, GfxMpsrtStorage::Alias);
  EXPECT_STREQ(gfx_mpsrt_storage_bridge_direction_name(matrix_bridge.direction),
               "alias");
}

TEST(GfxStagePolicyTest,
     MpsrtMatrixTensorDescriptorFlattensBatchToMatrixCount) {
  const auto desc = gfx_mpsrt_make_tensor_desc({2, 128, 64}, ov::element::f32,
                                               GfxStageStorageKind::Matrix);

  EXPECT_EQ(desc.dtype, GfxMpsrtDType::F32);
  EXPECT_EQ(desc.storage, GfxMpsrtStorage::Matrix);
  EXPECT_EQ(desc.layout, GfxMpsrtLayout::RowMajor);
  EXPECT_EQ(desc.matrix_count, 2u);
  EXPECT_EQ(desc.matrix_rows, 128u);
  EXPECT_EQ(desc.matrix_columns, 64u);
  EXPECT_EQ(desc.matrix_row_bytes, 64u * 4u);
  EXPECT_EQ(desc.byte_length, 2u * 128u * 64u * 4u);
}

TEST(GfxStagePolicyTest,
     MpsrtUnsignedIntegerDescriptorKeepsDTypeAndByteLength) {
  const auto desc = gfx_mpsrt_make_tensor_desc({4, 8}, ov::element::u32,
                                               GfxStageStorageKind::Buffer);

  EXPECT_EQ(desc.dtype, GfxMpsrtDType::U32);
  EXPECT_EQ(gfx_mpsrt_dtype_from_name(gfx_mpsrt_dtype_name(desc.dtype)),
            GfxMpsrtDType::U32);
  EXPECT_EQ(desc.storage, GfxMpsrtStorage::Buffer);
  EXPECT_EQ(desc.layout, GfxMpsrtLayout::Linear);
  EXPECT_EQ(desc.byte_length, 4u * 8u * 4u);
}

TEST(GfxStagePolicyTest,
     MpsrtTensorAbiDescriptorRoundTripsWithoutCppContainers) {
  auto desc = gfx_mpsrt_make_tensor_desc({2, 128, 64}, ov::element::f16,
                                         GfxStageStorageKind::Matrix,
                                         GfxMpsrtTensorFlagExternalIo);
  desc.byte_offset = 256;
  const auto abi = gfx_mpsrt_to_abi_desc(desc);
  const auto restored = gfx_mpsrt_from_abi_desc(abi);

  EXPECT_EQ(abi.dtype, static_cast<uint32_t>(GfxMpsrtDType::F16));
  EXPECT_EQ(abi.storage, static_cast<uint32_t>(GfxMpsrtStorage::Matrix));
  EXPECT_EQ(abi.layout, static_cast<uint32_t>(GfxMpsrtLayout::RowMajor));
  EXPECT_EQ(abi.rank, 3u);
  EXPECT_EQ(abi.dims[0], 2u);
  EXPECT_EQ(abi.dims[1], 128u);
  EXPECT_EQ(abi.dims[2], 64u);
  EXPECT_EQ(abi.strides[0], 128 * 64);
  EXPECT_EQ(abi.byte_offset, 256u);
  EXPECT_EQ(abi.byte_length, 2u * 128u * 64u * 2u);
  EXPECT_EQ(restored.dtype, desc.dtype);
  EXPECT_EQ(restored.storage, desc.storage);
  EXPECT_EQ(restored.layout, desc.layout);
  EXPECT_EQ(restored.matrix_rows, 128u);
  EXPECT_EQ(restored.matrix_columns, 64u);
  EXPECT_EQ(restored.flags, GfxMpsrtTensorFlagExternalIo);
}

TEST(GfxStagePolicyTest, MpsrtStageRecordKeyIsStableForMetalMatMul) {
  const auto matmul = make_matmul_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "MatMul", matmul, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  const auto desc = gfx_mpsrt_make_stage_desc(plan, "MatMul");

  EXPECT_EQ(desc.kind, GfxMpsrtStageKind::MPSGemm);
  ASSERT_TRUE(desc.stage_manifest.valid);
  EXPECT_EQ(desc.stage_manifest.stage_family, GfxKernelStageFamily::Gemm);
  EXPECT_EQ(desc.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMps);
  EXPECT_EQ(desc.stage_manifest.execution_kind,
            GfxKernelExecutionKind::VendorPrimitive);
  EXPECT_EQ(desc.stage_manifest.storage, GfxKernelStorageKind::Matrix);
  EXPECT_FALSE(desc.stage_manifest.custom_kernel.valid);
  EXPECT_EQ(desc.kernel_name, "mps_gemm");
  EXPECT_EQ(desc.builder_symbol, "ovgfx_mpsrt_encode_gemm");
  EXPECT_EQ(gfx_mpsrt_stage_record_key(desc),
            "mps_gemm|apple_mps|matrix|matrix|row_major|MatMul|apple_mps:"
            "matrix:MatMul");
}

TEST(GfxStagePolicyTest, MpsrtStageRecordKeyUsesNhwc4ForMetalConvolutionStage) {
  const auto conv = make_pointwise_conv_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "Convolution", conv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  const auto desc = gfx_mpsrt_make_stage_desc(plan, "Convolution");

  EXPECT_EQ(desc.kind, GfxMpsrtStageKind::MPSConv2D);
  ASSERT_TRUE(desc.stage_manifest.valid);
  EXPECT_EQ(desc.stage_manifest.stage_family,
            GfxKernelStageFamily::Convolution);
  EXPECT_EQ(desc.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMps);
  EXPECT_EQ(desc.stage_manifest.execution_kind,
            GfxKernelExecutionKind::VendorPrimitive);
  EXPECT_EQ(desc.stage_manifest.storage, GfxKernelStorageKind::Image);
  EXPECT_FALSE(desc.stage_manifest.custom_kernel.valid);
  EXPECT_EQ(desc.layout, GfxMpsrtLayout::NHWC4);
  EXPECT_EQ(desc.builder_symbol, "ovgfx_mpsrt_encode_conv2d");
  EXPECT_EQ(
      gfx_mpsrt_stage_kind_from_name(gfx_mpsrt_stage_kind_name(desc.kind)),
      GfxMpsrtStageKind::MPSConv2D);
  EXPECT_EQ(gfx_mpsrt_stage_record_key(desc),
            "mps_conv2d|apple_mps|image|image|nhwc4|Convolution|apple_mps:"
            "image:Convolution");
}

TEST(GfxStagePolicyTest, MpsrtConv2DStageRecordKeyCarriesFusedActivation) {
  const auto conv = make_pointwise_conv_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "Convolution", conv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  auto desc = gfx_mpsrt_make_stage_desc(plan, "Convolution");
  desc.conv2d_desc.fused_activation = 1u;

  EXPECT_EQ(gfx_mpsrt_stage_record_key(desc),
            "mps_conv2d|apple_mps|image|image|nhwc4|Convolution|apple_mps:"
            "image:Convolution|"
            "conv2d:g1:s1x1:d1x1:p0,0,0,0:act1");
}

TEST(GfxStagePolicyTest,
     MpsrtConv2DBuilderPlanCarriesVendorPrimitiveDescriptor) {
  const auto conv = make_light_spatial3x3_conv_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "Convolution", conv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  auto stage = gfx_mpsrt_make_stage_desc(plan, "Convolution");
  stage.conv2d_desc.groups = 1;
  stage.conv2d_desc.strides[0] = 1;
  stage.conv2d_desc.strides[1] = 1;
  stage.conv2d_desc.dilations[0] = 1;
  stage.conv2d_desc.dilations[1] = 1;
  stage.conv2d_desc.pads[0] = 1;
  stage.conv2d_desc.pads[1] = 1;
  stage.conv2d_desc.pads[2] = 1;
  stage.conv2d_desc.pads[3] = 1;

  const auto input = gfx_mpsrt_make_tensor_desc(
      {1, 16, 160, 160}, ov::element::f16, GfxStageStorageKind::Image,
      GfxMpsrtTensorFlagExternalIo);
  const auto weights = gfx_mpsrt_make_tensor_desc(
      {8, 16, 3, 3}, ov::element::f16, GfxStageStorageKind::Image,
      GfxMpsrtTensorFlagConst);
  const auto output = gfx_mpsrt_make_tensor_desc(
      {1, 8, 160, 160}, ov::element::f16, GfxStageStorageKind::Image,
      GfxMpsrtTensorFlagTransient);
  const auto record_key = gfx_mpsrt_stage_record_key(stage);
  EXPECT_NE(record_key.find("|conv2d:g1:s1x1:d1x1:p1,1,1,1"),
            std::string::npos);

  const auto builder_plan = gfx_mpsrt_make_builder_plan(stage, {input, weights},
                                                        {output}, record_key);
  ASSERT_TRUE(builder_plan.valid);
  ASSERT_EQ(builder_plan.records.size(), 5u);
  ASSERT_EQ(builder_plan.storage_bridges.size(), 2u);
  EXPECT_EQ(builder_plan.storage_bridges[0].value, 0u);
  EXPECT_EQ(builder_plan.storage_bridges[0].direction,
            GfxMpsrtStorageBridgeDirection::BufferToImage);
  EXPECT_EQ(builder_plan.storage_bridges[1].value, 2u);
  EXPECT_EQ(builder_plan.storage_bridges[1].direction,
            GfxMpsrtStorageBridgeDirection::ImageToBuffer);
  EXPECT_EQ(builder_plan.records[3].stage_kind, GfxMpsrtStageKind::MPSConv2D);
  EXPECT_EQ(builder_plan.records[3].conv2d_desc.pads[0], 1u);
  EXPECT_EQ(builder_plan.records[3].conv2d_desc.pads[3], 1u);

  runtime_mpsrt::MpsrtModel model;
  std::string error;
  ASSERT_TRUE(runtime_mpsrt::build_mpsrt_model_from_builder_plan(
      builder_plan, model, &error))
      << error;
  ASSERT_EQ(model.stages.size(), 1u);
  ASSERT_EQ(model.storage_bridges.size(), 2u);
  EXPECT_EQ(model.storage_bridges[0].value, 0u);
  EXPECT_EQ(model.storage_bridges[0].source_storage, GfxMpsrtStorage::Buffer);
  EXPECT_EQ(model.storage_bridges[0].target_storage, GfxMpsrtStorage::Image);
  EXPECT_EQ(model.storage_bridges[1].value, 2u);
  EXPECT_EQ(model.storage_bridges[1].source_storage, GfxMpsrtStorage::Image);
  EXPECT_EQ(model.storage_bridges[1].target_storage, GfxMpsrtStorage::Buffer);
  EXPECT_EQ(model.stages.front().kind, GfxMpsrtStageKind::MPSConv2D);
  EXPECT_EQ(model.stages.front().conv2d_desc.pads[0], 1u);
  EXPECT_EQ(model.stages.front().conv2d_desc.pads[3], 1u);
  EXPECT_EQ(runtime_mpsrt::mpsrt_model_resource_lifetime_count(
                model, runtime_mpsrt::MpsrtRuntimeResourceLifetime::External),
            2u);
  EXPECT_EQ(runtime_mpsrt::mpsrt_model_resource_lifetime_count(
                model, runtime_mpsrt::MpsrtRuntimeResourceLifetime::Model),
            1u);
  const auto *weight_resource =
      runtime_mpsrt::find_mpsrt_resource_for_value(model, 1u);
  ASSERT_NE(weight_resource, nullptr);
  EXPECT_EQ(weight_resource->role, GfxMpsrtExternalBufferRole::ConstBuffer);
  EXPECT_EQ(weight_resource->lifetime,
            runtime_mpsrt::MpsrtRuntimeResourceLifetime::Model);
  EXPECT_TRUE((weight_resource->tensor_desc.flags & GfxMpsrtTensorFlagConst) !=
              0);
  std::vector<runtime_mpsrt::MpsrtTensorBindingPlanEntry> binding_plan;
  ASSERT_TRUE(runtime_mpsrt::mpsrt_model_tensor_binding_plan(
      model, binding_plan, &error))
      << error;
  ASSERT_EQ(binding_plan.size(), 3u);
  EXPECT_EQ(binding_plan[0].lifetime,
            runtime_mpsrt::MpsrtRuntimeResourceLifetime::External);
  EXPECT_EQ(binding_plan[0].value, 0u);
  EXPECT_EQ(binding_plan[0].bridge_direction,
            GfxMpsrtStorageBridgeDirection::BufferToImage);
  EXPECT_EQ(binding_plan[1].lifetime,
            runtime_mpsrt::MpsrtRuntimeResourceLifetime::External);
  EXPECT_EQ(binding_plan[1].value, 2u);
  EXPECT_EQ(binding_plan[1].bridge_direction,
            GfxMpsrtStorageBridgeDirection::ImageToBuffer);
  EXPECT_EQ(binding_plan[2].lifetime,
            runtime_mpsrt::MpsrtRuntimeResourceLifetime::Model);
  EXPECT_EQ(binding_plan[2].value, 1u);
}

TEST(GfxStagePolicyTest,
     MpsrtPool2DBuilderPlanCarriesVendorPrimitiveDescriptor) {
  const auto pool = make_aligned_maxpool_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "MaxPool", pool, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  auto stage = gfx_mpsrt_make_stage_desc(plan, "MaxPool");
  stage.pool2d_desc.is_avg = 0;
  stage.pool2d_desc.kernel[0] = 2;
  stage.pool2d_desc.kernel[1] = 2;
  stage.pool2d_desc.strides[0] = 2;
  stage.pool2d_desc.strides[1] = 2;

  const auto input = gfx_mpsrt_make_tensor_desc(
      {1, 4, 16, 16}, ov::element::f16, GfxStageStorageKind::Image,
      GfxMpsrtTensorFlagExternalIo);
  const auto output = gfx_mpsrt_make_tensor_desc({1, 4, 8, 8}, ov::element::f16,
                                                 GfxStageStorageKind::Image,
                                                 GfxMpsrtTensorFlagTransient);
  const auto record_key = gfx_mpsrt_stage_record_key(stage);
  EXPECT_NE(record_key.find("|pool2d:max:k2x2:s2x2:d1x1:p0,0,0,0"),
            std::string::npos);

  const auto builder_plan =
      gfx_mpsrt_make_builder_plan(stage, {input}, {output}, record_key);
  ASSERT_TRUE(builder_plan.valid);
  ASSERT_EQ(builder_plan.records.size(), 4u);
  EXPECT_EQ(builder_plan.records[2].stage_kind, GfxMpsrtStageKind::MPSPool2D);
  EXPECT_EQ(builder_plan.records[2].pool2d_desc.kernel[0], 2u);
  EXPECT_EQ(builder_plan.records[2].pool2d_desc.strides[1], 2u);

  runtime_mpsrt::MpsrtModel model;
  std::string error;
  ASSERT_TRUE(runtime_mpsrt::build_mpsrt_model_from_builder_plan(
      builder_plan, model, &error))
      << error;
  ASSERT_EQ(model.stages.size(), 1u);
  EXPECT_EQ(model.stages.front().kind, GfxMpsrtStageKind::MPSPool2D);
  EXPECT_EQ(model.stages.front().pool2d_desc.kernel[0], 2u);
  EXPECT_EQ(model.stages.front().pool2d_desc.strides[1], 2u);
}

TEST(GfxStagePolicyTest,
     MpsrtResize2DBuilderPlanCarriesVendorPrimitiveDescriptor) {
  GfxStageOptimizationPlan plan{};
  plan.placement.domain = GfxStageBackendDomain::AppleMps;
  plan.placement.storage = GfxStageStorageKind::Image;
  plan.placement.uses_vendor_primitive = true;
  plan.placement.uses_custom_kernel = false;
  plan.placement.specialization_key = "apple_mps:image:Interpolate";
  auto stage = gfx_mpsrt_make_stage_desc(plan, "Interpolate");
  stage.resize2d_desc.nearest = 0;
  stage.resize2d_desc.align_corners = 0;
  stage.resize2d_desc.half_pixel_centers = 1;

  const auto input = gfx_mpsrt_make_tensor_desc(
      {1, 4, 16, 16}, ov::element::f16, GfxStageStorageKind::Image,
      GfxMpsrtTensorFlagExternalIo);
  const auto output = gfx_mpsrt_make_tensor_desc(
      {1, 4, 32, 32}, ov::element::f16, GfxStageStorageKind::Image,
      GfxMpsrtTensorFlagTransient);
  const auto record_key = gfx_mpsrt_stage_record_key(stage);
  EXPECT_EQ(record_key.find("|resize2d:"), std::string::npos);

  const auto builder_plan =
      gfx_mpsrt_make_builder_plan(stage, {input}, {output}, record_key);
  ASSERT_TRUE(builder_plan.valid);
  ASSERT_EQ(builder_plan.records.size(), 4u);
  EXPECT_EQ(builder_plan.records[2].stage_kind, GfxMpsrtStageKind::MPSResize2D);
  EXPECT_EQ(builder_plan.records[2].resize2d_desc.nearest, 0u);
  EXPECT_EQ(builder_plan.records[2].resize2d_desc.half_pixel_centers, 1u);

  runtime_mpsrt::MpsrtModel model;
  std::string error;
  ASSERT_TRUE(runtime_mpsrt::build_mpsrt_model_from_builder_plan(
      builder_plan, model, &error))
      << error;
  ASSERT_EQ(model.stages.size(), 1u);
  EXPECT_EQ(model.stages.front().kind, GfxMpsrtStageKind::MPSResize2D);
  EXPECT_EQ(model.stages.front().resize2d_desc.nearest, 0u);
  ASSERT_EQ(model.stages.front().output_descs.size(), 1u);
  EXPECT_EQ(model.stages.front().output_descs.front().image_width, 32u);
}

TEST(GfxStagePolicyTest,
     MetalBilinearInterpolateSourcePlanUsesMpsResize2DInsteadOfMslFallback) {
  const auto resize = make_bilinear_interpolate_node();
  auto &ctx = gfx_mlir_context();
  auto module = build_mlir_for_node(resize, ctx);
  ASSERT_TRUE(module);

  KernelSource source;
  source.module = module;
  source.entry_point = "interpolate_kernel";
  const auto source_plan = configure_apple_metal_kernel_source_plan_for_stage(
      source, resize, nullptr, "Interpolate",
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, ActivationKind::Identity, ov::element::f16,
      /*has_runtime_slice_params=*/false);
  ASSERT_TRUE(source_plan.valid());
  EXPECT_TRUE(source_plan.requires_mpsrt_model);
  EXPECT_EQ(source_plan.first_stage_kind, GfxMpsrtStageKind::MPSResize2D);
  EXPECT_EQ(source_plan.last_stage_kind, GfxMpsrtStageKind::MPSResize2D);
  EXPECT_TRUE(gfx_mpsrt_source_plan_is_io_only_apple_mps_vendor(source_plan));
  EXPECT_TRUE(source_plan.has_runtime_binding);
  EXPECT_EQ(source_plan.runtime_binding.inputs, std::vector<size_t>({0}));
  EXPECT_EQ(source_plan.runtime_binding.input_arg_count, 1u);
  EXPECT_EQ(source_plan.source.signature.arg_count, 2u);
  EXPECT_EQ(source_plan.source.signature.output_arg_count, 1u);
  EXPECT_TRUE(source_plan.source.msl_source.empty());
  EXPECT_FALSE(static_cast<bool>(source_plan.source.msl_generator));

  GfxMpsrtProgram program;
  ASSERT_TRUE(read_module_mpsrt_program(module, program));
  ASSERT_TRUE(program.external_buffer_abi.valid);
  EXPECT_EQ(program.external_buffer_abi.buffer_roles,
            std::vector<GfxMpsrtExternalBufferRole>(
                {GfxMpsrtExternalBufferRole::TensorInput,
                 GfxMpsrtExternalBufferRole::TensorOutput}));
  ASSERT_EQ(program.stages.size(), 1u);
  EXPECT_EQ(program.stages.front().stage.kind, GfxMpsrtStageKind::MPSResize2D);
  EXPECT_EQ(program.stages.front().stage.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMps);
  EXPECT_EQ(program.stages.front().stage.stage_manifest.execution_kind,
            GfxKernelExecutionKind::VendorPrimitive);
}

TEST(GfxStagePolicyTest,
     MetalMaxPoolSourcePlanUsesMpsPool2DInsteadOfMslFallback) {
  const auto pool = make_aligned_maxpool_node();
  auto &ctx = gfx_mlir_context();
  auto module = build_mlir_for_node(pool, ctx);
  ASSERT_TRUE(module);

  KernelSource source;
  source.module = module;
  source.entry_point = "pool2d_kernel";
  const auto source_plan = configure_apple_metal_kernel_source_plan_for_stage(
      source, pool, nullptr, "MaxPool",
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, ActivationKind::Identity, ov::element::f16,
      /*has_runtime_slice_params=*/false);
  ASSERT_TRUE(source_plan.valid());
  EXPECT_TRUE(source_plan.requires_mpsrt_model);
  EXPECT_EQ(source_plan.first_stage_kind, GfxMpsrtStageKind::MPSPool2D);
  EXPECT_EQ(source_plan.last_stage_kind, GfxMpsrtStageKind::MPSPool2D);
  EXPECT_TRUE(gfx_mpsrt_source_plan_is_io_only_apple_mps_vendor(source_plan));
  EXPECT_TRUE(source_plan.has_runtime_binding);
  EXPECT_EQ(source_plan.runtime_binding.inputs, std::vector<size_t>({0, 1}));
  EXPECT_EQ(source_plan.runtime_binding.input_arg_count, 2u);
  EXPECT_TRUE(source_plan.source.msl_source.empty());
  EXPECT_FALSE(static_cast<bool>(source_plan.source.msl_generator));

  GfxMpsrtProgram program;
  ASSERT_TRUE(read_module_mpsrt_program(module, program));
  ASSERT_TRUE(program.external_buffer_abi.valid);
  EXPECT_EQ(program.external_buffer_abi.buffer_roles,
            std::vector<GfxMpsrtExternalBufferRole>(
                {GfxMpsrtExternalBufferRole::TensorInput,
                 GfxMpsrtExternalBufferRole::RuntimeParams,
                 GfxMpsrtExternalBufferRole::TensorOutput}));
  ASSERT_EQ(program.stages.size(), 1u);
  EXPECT_EQ(program.stages.front().stage.kind, GfxMpsrtStageKind::MPSPool2D);
  EXPECT_EQ(program.stages.front().stage.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMps);
  EXPECT_EQ(program.stages.front().stage.stage_manifest.execution_kind,
            GfxKernelExecutionKind::VendorPrimitive);
}

TEST(GfxStagePolicyTest,
     MetalSoftmaxSourcePlanUsesMpsSoftmaxInsteadOfMslFallback) {
  const auto softmax = make_last_dim_softmax_node();
  auto &ctx = gfx_mlir_context();
  auto module = build_mlir_for_node(softmax, ctx);
  ASSERT_TRUE(module);

  KernelSource source;
  source.module = module;
  source.entry_point = "softmax_kernel";
  const auto source_plan = configure_apple_metal_kernel_source_plan_for_stage(
      source, softmax, nullptr, "Softmax",
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, ActivationKind::Identity, ov::element::f16,
      /*has_runtime_slice_params=*/false);
  ASSERT_TRUE(source_plan.valid());
  EXPECT_TRUE(source_plan.requires_mpsrt_model);
  EXPECT_EQ(source_plan.first_stage_kind, GfxMpsrtStageKind::MPSSoftmax);
  EXPECT_EQ(source_plan.last_stage_kind, GfxMpsrtStageKind::MPSSoftmax);
  EXPECT_TRUE(gfx_mpsrt_source_plan_is_io_only_apple_mps_vendor(source_plan));
  EXPECT_TRUE(source_plan.has_runtime_binding);
  EXPECT_EQ(source_plan.runtime_binding.inputs, std::vector<size_t>({0}));
  EXPECT_EQ(source_plan.runtime_binding.input_arg_count, 1u);
  EXPECT_TRUE(source_plan.source.msl_source.empty());
  EXPECT_FALSE(static_cast<bool>(source_plan.source.msl_generator));

  GfxMpsrtProgram program;
  ASSERT_TRUE(read_module_mpsrt_program(module, program));
  ASSERT_TRUE(program.external_buffer_abi.valid);
  EXPECT_EQ(program.external_buffer_abi.buffer_roles,
            std::vector<GfxMpsrtExternalBufferRole>(
                {GfxMpsrtExternalBufferRole::TensorInput,
                 GfxMpsrtExternalBufferRole::TensorOutput}));
  ASSERT_EQ(program.stages.size(), 1u);
  EXPECT_EQ(program.stages.front().stage.kind, GfxMpsrtStageKind::MPSSoftmax);
  EXPECT_EQ(program.stages.front().stage.softmax_desc.axis, 2u);
  EXPECT_EQ(program.stages.front().stage.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMps);
  EXPECT_EQ(program.stages.front().stage.stage_manifest.execution_kind,
            GfxKernelExecutionKind::VendorPrimitive);
}

TEST(GfxStagePolicyTest, MetalTopKSourcePlanUsesMpsTopKInsteadOfMslFallback) {
  const auto topk = make_last_dim_topk_node();
  auto &ctx = gfx_mlir_context();
  auto module = build_mlir_for_node(topk, ctx);
  ASSERT_TRUE(module);

  KernelSource source;
  source.module = module;
  source.entry_point = "topk_kernel";
  const auto source_plan = configure_apple_metal_kernel_source_plan_for_stage(
      source, topk, nullptr, "TopK",
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, ActivationKind::Identity, ov::element::f16,
      /*has_runtime_slice_params=*/false);
  ASSERT_TRUE(source_plan.valid());
  EXPECT_TRUE(source_plan.requires_mpsrt_model);
  EXPECT_EQ(source_plan.first_stage_kind, GfxMpsrtStageKind::MPSTopK);
  EXPECT_EQ(source_plan.last_stage_kind, GfxMpsrtStageKind::MPSTopK);
  EXPECT_TRUE(gfx_mpsrt_source_plan_is_io_only_apple_mps_vendor(source_plan));
  EXPECT_TRUE(source_plan.has_runtime_binding);
  EXPECT_EQ(source_plan.runtime_binding.inputs, std::vector<size_t>({0}));
  EXPECT_EQ(source_plan.runtime_binding.input_arg_count, 1u);
  EXPECT_TRUE(source_plan.source.msl_source.empty());
  EXPECT_FALSE(static_cast<bool>(source_plan.source.msl_generator));

  GfxMpsrtProgram program;
  ASSERT_TRUE(read_module_mpsrt_program(module, program));
  ASSERT_TRUE(program.external_buffer_abi.valid);
  EXPECT_EQ(program.external_buffer_abi.buffer_roles,
            std::vector<GfxMpsrtExternalBufferRole>(
                {GfxMpsrtExternalBufferRole::TensorInput,
                 GfxMpsrtExternalBufferRole::TensorOutput,
                 GfxMpsrtExternalBufferRole::TensorOutput}));
  ASSERT_EQ(program.stages.size(), 1u);
  EXPECT_EQ(program.stages.front().stage.kind, GfxMpsrtStageKind::MPSTopK);
  EXPECT_EQ(program.stages.front().stage.topk_desc.k, 4u);
  EXPECT_EQ(program.stages.front().stage.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMps);
  EXPECT_EQ(program.stages.front().stage.stage_manifest.execution_kind,
            GfxKernelExecutionKind::VendorPrimitive);
}

TEST(GfxStagePolicyTest,
     MetalSoftmaxSourcePlanRebuildsCleanMpsModuleWhenInputModuleIsAlreadyMsl) {
  const auto softmax = make_last_dim_softmax_node();
  auto &ctx = gfx_mlir_context();
  auto msl_module = build_mlir_for_node(softmax, ctx);
  ASSERT_TRUE(msl_module);

  GfxStageOptimizationPlan msl_plan{};
  msl_plan.placement.domain = GfxStageBackendDomain::AppleMsl;
  msl_plan.placement.storage = GfxStageStorageKind::Buffer;
  msl_plan.placement.uses_custom_kernel = true;
  msl_plan.placement.specialization_key = "apple_msl:buffer:Softmax";
  annotate_msl_module_with_stage_plan(msl_module, msl_plan, "Softmax",
                                      "softmax_kernel");

  GfxMpsrtModuleStagePlan original_stage_plan{};
  ASSERT_TRUE(read_module_mpsrt_stage_plan(msl_module, original_stage_plan));
  ASSERT_EQ(original_stage_plan.stage.kind, GfxMpsrtStageKind::MSLDispatch);
  ASSERT_EQ(original_stage_plan.stage.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMsl);

  KernelSource source;
  source.module = msl_module;
  source.entry_point = "softmax_kernel";
  const auto source_plan = configure_apple_metal_kernel_source_plan_for_stage(
      source, softmax, nullptr, "Softmax",
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, ActivationKind::Identity, ov::element::f16,
      /*has_runtime_slice_params=*/false);
  ASSERT_TRUE(source_plan.valid());
  auto vendor_module = source_plan.source.module;
  EXPECT_NE(vendor_module.getOperation(), msl_module.getOperation());
  EXPECT_TRUE(source_plan.requires_mpsrt_model);
  EXPECT_EQ(source_plan.first_stage_kind, GfxMpsrtStageKind::MPSSoftmax);
  EXPECT_EQ(source_plan.last_stage_kind, GfxMpsrtStageKind::MPSSoftmax);
  EXPECT_TRUE(gfx_mpsrt_source_plan_is_io_only_apple_mps_vendor(source_plan));
  EXPECT_TRUE(source_plan.has_runtime_binding);
  EXPECT_EQ(source_plan.runtime_binding.inputs, std::vector<size_t>({0}));
  EXPECT_EQ(source_plan.runtime_binding.input_arg_count, 1u);
  EXPECT_TRUE(source_plan.source.msl_source.empty());
  EXPECT_FALSE(static_cast<bool>(source_plan.source.msl_generator));

  GfxMpsrtProgram program;
  ASSERT_TRUE(read_module_mpsrt_program(vendor_module, program));
  ASSERT_TRUE(program.external_buffer_abi.valid);
  EXPECT_EQ(program.external_buffer_abi.buffer_roles,
            std::vector<GfxMpsrtExternalBufferRole>(
                {GfxMpsrtExternalBufferRole::TensorInput,
                 GfxMpsrtExternalBufferRole::TensorOutput}));
  ASSERT_EQ(program.stages.size(), 1u);
  EXPECT_EQ(program.stages.front().stage.kind, GfxMpsrtStageKind::MPSSoftmax);
  EXPECT_EQ(program.stages.front().stage.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMps);
  EXPECT_EQ(program.stages.front().stage.stage_manifest.execution_kind,
            GfxKernelExecutionKind::VendorPrimitive);

  ASSERT_TRUE(materialize_gfx_apple_mpsrt_runtime_abi_call_plan(vendor_module));
  GfxMpsrtBuilderPlan builder_plan{};
  ASSERT_TRUE(
      read_gfx_apple_mpsrt_runtime_abi_call_plan(vendor_module, builder_plan));
  ASSERT_EQ(builder_plan.records.size(), 4u);
  ASSERT_EQ(builder_plan.records[1].tensor_descs.size(), 1u);
  ASSERT_EQ(builder_plan.records[2].tensor_descs.size(), 1u);
  EXPECT_EQ(builder_plan.records[1].tensor_descs.front().storage,
            static_cast<uint32_t>(GfxMpsrtStorage::Matrix));
  EXPECT_EQ(builder_plan.records[2].tensor_descs.front().storage,
            static_cast<uint32_t>(GfxMpsrtStorage::Matrix));

  runtime_mpsrt::MpsrtModel model;
  std::string error;
  ASSERT_TRUE(runtime_mpsrt::build_mpsrt_model_from_builder_plan(
      builder_plan, model, &error))
      << error;
  ASSERT_TRUE(
      runtime_mpsrt::adapt_mpsrt_model_to_external_buffer_abi(
          model, source_plan.source.signature.arg_count,
          source_plan.source.signature.output_arg_count, &error))
      << error;
  ASSERT_EQ(model.stages.size(), 1u);
  ASSERT_EQ(model.stages.front().output_descs.size(), 1u);
  const auto *input_tensor =
      runtime_mpsrt::find_mpsrt_resource_for_value(
          model, model.stages.front().inputs.front());
  ASSERT_NE(input_tensor, nullptr);
  EXPECT_EQ(input_tensor->tensor_desc.storage,
            static_cast<uint32_t>(GfxMpsrtStorage::Matrix));
  EXPECT_EQ(model.stages.front().output_descs.front().storage,
            static_cast<uint32_t>(GfxMpsrtStorage::Matrix));
  EXPECT_NE(input_tensor->tensor_desc.matrix_rows, 0u);
  EXPECT_NE(model.stages.front().output_descs.front().matrix_rows, 0u);

  GfxMpsrtModuleStagePlan fallback_stage_plan{};
  ASSERT_TRUE(read_module_mpsrt_stage_plan(msl_module, fallback_stage_plan));
  EXPECT_EQ(fallback_stage_plan.stage.kind, GfxMpsrtStageKind::MSLDispatch);
  EXPECT_EQ(fallback_stage_plan.stage.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMsl);
}

TEST(GfxStagePolicyTest,
     MpsrtSoftmaxBuilderPlanCarriesVendorPrimitiveDescriptor) {
  const auto softmax = make_last_dim_softmax_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "Softmax", softmax, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  auto stage = gfx_mpsrt_make_stage_desc(plan, "Softmax");
  stage.softmax_desc.axis = 2;
  stage.softmax_desc.log_softmax = 0;

  const auto input = gfx_mpsrt_make_tensor_desc({2, 8, 16}, ov::element::f16,
                                                GfxStageStorageKind::Matrix,
                                                GfxMpsrtTensorFlagExternalIo);
  const auto output = gfx_mpsrt_make_tensor_desc({2, 8, 16}, ov::element::f16,
                                                 GfxStageStorageKind::Matrix,
                                                 GfxMpsrtTensorFlagTransient);
  const auto record_key = gfx_mpsrt_stage_record_key(stage);
  EXPECT_NE(record_key.find("|softmax:axis2"), std::string::npos);

  const auto builder_plan =
      gfx_mpsrt_make_builder_plan(stage, {input}, {output}, record_key);
  ASSERT_TRUE(builder_plan.valid);
  ASSERT_EQ(builder_plan.records.size(), 4u);
  EXPECT_EQ(builder_plan.records[2].stage_kind, GfxMpsrtStageKind::MPSSoftmax);
  EXPECT_EQ(builder_plan.records[2].softmax_desc.axis, 2u);
  EXPECT_EQ(builder_plan.records[2].softmax_desc.log_softmax, 0u);

  runtime_mpsrt::MpsrtModel model;
  std::string error;
  ASSERT_TRUE(runtime_mpsrt::build_mpsrt_model_from_builder_plan(
      builder_plan, model, &error))
      << error;
  ASSERT_EQ(model.stages.size(), 1u);
  EXPECT_EQ(model.stages.front().kind, GfxMpsrtStageKind::MPSSoftmax);
  EXPECT_EQ(model.stages.front().softmax_desc.axis, 2u);
  EXPECT_EQ(model.stages.front().softmax_desc.log_softmax, 0u);
}

TEST(GfxStagePolicyTest, MpsrtTopKBuilderPlanCarriesVendorPrimitiveDescriptor) {
  const auto topk = make_last_dim_topk_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "TopK", topk, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  auto stage = gfx_mpsrt_make_stage_desc(plan, "TopK");
  stage.topk_desc.axis = 2;
  stage.topk_desc.k = 4;
  stage.topk_desc.mode_max = 1;
  stage.topk_desc.sort_type = 1u;

  const auto input = gfx_mpsrt_make_tensor_desc({2, 8, 16}, ov::element::f16,
                                                GfxStageStorageKind::Matrix,
                                                GfxMpsrtTensorFlagExternalIo);
  const auto output_values = gfx_mpsrt_make_tensor_desc(
      {2, 8, 4}, ov::element::f16, GfxStageStorageKind::Matrix,
      GfxMpsrtTensorFlagTransient);
  const auto output_indices = gfx_mpsrt_make_tensor_desc(
      {2, 8, 4}, ov::element::i32, GfxStageStorageKind::Matrix,
      GfxMpsrtTensorFlagTransient);
  const auto record_key = gfx_mpsrt_stage_record_key(stage);
  EXPECT_NE(record_key.find("|topk:axis2:k4:max:sort1"), std::string::npos);

  const auto builder_plan = gfx_mpsrt_make_builder_plan(
      stage, {input}, {output_values, output_indices}, record_key);
  ASSERT_TRUE(builder_plan.valid);
  ASSERT_EQ(builder_plan.records.size(), 4u);
  EXPECT_EQ(builder_plan.records[2].stage_kind, GfxMpsrtStageKind::MPSTopK);
  EXPECT_EQ(builder_plan.records[2].topk_desc.axis, 2u);
  EXPECT_EQ(builder_plan.records[2].topk_desc.k, 4u);
  EXPECT_EQ(builder_plan.records[2].topk_desc.mode_max, 1u);

  runtime_mpsrt::MpsrtModel model;
  std::string error;
  ASSERT_TRUE(runtime_mpsrt::build_mpsrt_model_from_builder_plan(
      builder_plan, model, &error))
      << error;
  ASSERT_EQ(model.stages.size(), 1u);
  EXPECT_EQ(model.stages.front().kind, GfxMpsrtStageKind::MPSTopK);
  EXPECT_EQ(model.stages.front().topk_desc.axis, 2u);
  EXPECT_EQ(model.stages.front().topk_desc.k, 4u);
  ASSERT_EQ(model.stages.front().output_descs.size(), 2u);
  EXPECT_EQ(model.stages.front().output_descs[0].matrix_columns, 4u);
  EXPECT_EQ(model.stages.front().output_descs[1].dtype,
            static_cast<uint32_t>(GfxMpsrtDType::I32));
}

TEST(GfxStagePolicyTest, MpsrtStageRecordKeyKeepsElementwiseInMslDispatch) {
  const auto add = make_large_add_node();
  const auto plan = select_stage_optimization_plan(nullptr, GpuBackend::Metal,
                                                   "Add", add, ov::element::f16,
                                                   /*has_bias=*/false,
                                                   /*has_activation=*/false,
                                                   /*has_batchnorm=*/false, {});
  const auto desc = gfx_mpsrt_make_stage_desc(plan, "Add");

  EXPECT_EQ(desc.kind, GfxMpsrtStageKind::MSLDispatch);
  ASSERT_TRUE(desc.stage_manifest.valid);
  EXPECT_EQ(desc.stage_manifest.stage_family, GfxKernelStageFamily::Eltwise);
  EXPECT_EQ(desc.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMsl);
  EXPECT_EQ(desc.stage_manifest.execution_kind,
            GfxKernelExecutionKind::CustomKernel);
  EXPECT_EQ(desc.stage_manifest.storage, GfxKernelStorageKind::Buffer);
  ASSERT_TRUE(desc.stage_manifest.custom_kernel.valid);
  EXPECT_EQ(desc.stage_manifest.custom_kernel.kernel_family,
            "eltwise_fused_buffer");
  ASSERT_TRUE(desc.stage_manifest.custom_kernel.external_buffer_abi.valid);
  EXPECT_EQ(
      desc.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>(
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
           GfxKernelBufferRole::TensorOutput, GfxKernelBufferRole::ScalarParam,
           GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams}));
  EXPECT_EQ(desc.kernel_name, "Add");
  EXPECT_EQ(desc.dispatch_kernel_family, "eltwise_fused_buffer");
  EXPECT_EQ(desc.dispatch_entry_point, "eltwise_fused_buffer");
  EXPECT_EQ(desc.dispatch_kernel_family_id,
            static_cast<uint32_t>(GfxKernelFamily::EltwiseFusedBuffer));
  EXPECT_EQ(desc.dispatch_flags,
            GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
  EXPECT_EQ(desc.dispatch_threads_per_threadgroup, 256u);
  EXPECT_TRUE(desc.dispatch_precompiled_kernel_required);
  EXPECT_EQ(desc.builder_symbol, "ovgfx_mpsrt_encode_dispatch");
  EXPECT_EQ(
      gfx_mpsrt_stage_record_key(desc),
      "msl_dispatch|apple_msl|buffer|buffer|linear|Add|apple_msl:buffer:Add|"
      "dispatch:eltwise_fused_buffer:eltwise_fused_buffer:linear_1d:tg256:"
      "metallib");
}

TEST(GfxStagePolicyTest,
     MpsrtStageDescUsesExplicitMslEntryPointForManifestClassification) {
  const auto conv = make_pointwise_conv_node();
  auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "Convolution", conv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  plan.placement.domain = GfxStageBackendDomain::AppleMsl;
  plan.placement.storage = GfxStageStorageKind::Buffer;
  plan.placement.uses_vendor_primitive = false;
  plan.placement.uses_custom_kernel = true;
  plan.placement.specialization_key = "apple_msl:buffer:Convolution";

  const auto desc =
      gfx_mpsrt_make_stage_desc(plan, "Convolution", "conv3d_kernel");

  EXPECT_EQ(desc.kind, GfxMpsrtStageKind::MSLDispatch);
  ASSERT_TRUE(desc.stage_manifest.valid);
  EXPECT_EQ(desc.stage_manifest.stage_family, GfxKernelStageFamily::Conv3D);
  EXPECT_EQ(desc.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMsl);
  EXPECT_EQ(desc.stage_manifest.execution_kind,
            GfxKernelExecutionKind::CustomKernel);
  ASSERT_TRUE(desc.stage_manifest.custom_kernel.valid);
  EXPECT_EQ(desc.dispatch_kernel_family, "conv3d_direct_or_im2col");
  EXPECT_EQ(desc.dispatch_entry_point, "conv3d_direct_or_im2col");
  EXPECT_EQ(desc.dispatch_kernel_family_id,
            static_cast<uint32_t>(GfxKernelFamily::Conv3DDirectOrIm2col));
  EXPECT_EQ(gfx_mpsrt_stage_record_key(desc),
            "msl_dispatch|apple_msl|buffer|buffer|linear|Convolution|apple_msl:"
            "buffer:Convolution|"
            "dispatch:conv3d_direct_or_im2col:conv3d_direct_or_im2col:linear_"
            "1d:tg128:metallib");
}

TEST(GfxStagePolicyTest,
     CustomKernelStagePlanClassifiesConv2DKernelAsCustomConvolution) {
  const auto plan =
      make_gfx_custom_kernel_stage_plan("Convolution", "conv2d_kernel");

  ASSERT_TRUE(plan.valid);
  EXPECT_EQ(plan.family, GfxKernelFamily::Conv2DDirectOrIm2col);
  ASSERT_TRUE(plan.stage_manifest.valid);
  EXPECT_EQ(plan.stage_manifest.stage_family,
            GfxKernelStageFamily::Convolution);
  EXPECT_EQ(plan.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMsl);
  EXPECT_EQ(plan.stage_manifest.execution_kind,
            GfxKernelExecutionKind::CustomKernel);
  EXPECT_EQ(plan.stage_manifest.storage, GfxKernelStorageKind::Buffer);
  ASSERT_TRUE(plan.stage_manifest.custom_kernel.valid);
  EXPECT_EQ(plan.stage_manifest.custom_kernel.kernel_family,
            "conv2d_direct_or_im2col");
  EXPECT_EQ(plan.stage_manifest.custom_kernel.entry_point,
            "conv2d_direct_or_im2col");
  EXPECT_EQ(
      plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>(
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::ConstTensor,
           GfxKernelBufferRole::ConstTensor, GfxKernelBufferRole::ConstTensor,
           GfxKernelBufferRole::ConstTensor, GfxKernelBufferRole::ConstTensor,
           GfxKernelBufferRole::ConstTensor, GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::TensorOutput}));
}

TEST(GfxStagePolicyTest,
     CustomKernelStagePlanClassifiesMatMulKernelAsCustomGemm) {
  const auto plan =
      make_gfx_custom_kernel_stage_plan("MatMul", "matmul_kernel");

  ASSERT_TRUE(plan.valid);
  EXPECT_EQ(plan.family, GfxKernelFamily::MatMulBuffer);
  ASSERT_TRUE(plan.stage_manifest.valid);
  EXPECT_EQ(plan.stage_manifest.stage_family, GfxKernelStageFamily::Gemm);
  EXPECT_EQ(plan.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMsl);
  EXPECT_EQ(plan.stage_manifest.execution_kind,
            GfxKernelExecutionKind::CustomKernel);
  EXPECT_EQ(plan.stage_manifest.storage, GfxKernelStorageKind::Buffer);
  ASSERT_TRUE(plan.stage_manifest.custom_kernel.valid);
  EXPECT_EQ(plan.stage_manifest.custom_kernel.kernel_family, "matmul_buffer");
  EXPECT_EQ(plan.stage_manifest.custom_kernel.entry_point, "matmul_buffer");
  ASSERT_TRUE(plan.stage_manifest.custom_kernel.external_buffer_abi.valid);
  EXPECT_EQ(
      plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput}));
}

TEST(GfxStagePolicyTest,
     CustomKernelStagePlanClassifiesCompressedMatMulWithExplicitConstRoles) {
  const auto plan = make_gfx_custom_kernel_stage_plan(
      "CompressedMatMul", "compressed_matmul_kernel");

  ASSERT_TRUE(plan.valid);
  EXPECT_EQ(plan.family, GfxKernelFamily::MatMulBuffer);
  ASSERT_TRUE(plan.stage_manifest.custom_kernel.external_buffer_abi.valid);
  EXPECT_EQ(
      plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::ConstTensor,
                                        GfxKernelBufferRole::ConstTensor,
                                        GfxKernelBufferRole::TensorOutput}));

  const auto binding =
      make_msl_runtime_binding_plan_from_stage_manifest(plan.stage_manifest);
  ASSERT_TRUE(binding.valid());
  EXPECT_EQ(binding.runtime_binding.inputs, std::vector<size_t>({0}));
  EXPECT_EQ(binding.runtime_binding.input_arg_count, 3u);
  EXPECT_EQ(binding.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 1, 1}));
  EXPECT_EQ(binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3}));
}

TEST(GfxStagePolicyTest,
     CustomKernelStagePlanClassifiesSdpaKernelsWithRuntimeParamsTail) {
  const auto sdpa_plan = make_gfx_custom_kernel_stage_plan(
      "ScaledDotProductAttention", "sdpa_kernel");
  ASSERT_TRUE(sdpa_plan.valid);
  EXPECT_EQ(sdpa_plan.family, GfxKernelFamily::MaskedSoftmaxAttention);
  ASSERT_TRUE(sdpa_plan.stage_manifest.custom_kernel.external_buffer_abi.valid);
  EXPECT_EQ(
      sdpa_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>(
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
           GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::TensorOutput}));

  const auto sdpa_binding = make_msl_runtime_binding_plan_from_stage_manifest(
      sdpa_plan.stage_manifest);
  ASSERT_TRUE(sdpa_binding.valid());
  EXPECT_EQ(sdpa_binding.runtime_binding.inputs,
            std::vector<size_t>({0, 1, 2, 3}));
  EXPECT_EQ(sdpa_binding.runtime_binding.input_arg_count, 5u);
  EXPECT_EQ(sdpa_binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3, 4, 5}));

  const auto sdpa_nomask_plan = make_gfx_custom_kernel_stage_plan(
      "ScaledDotProductAttention", "sdpa_nomask_kernel");
  ASSERT_TRUE(sdpa_nomask_plan.valid);
  EXPECT_EQ(sdpa_nomask_plan.family, GfxKernelFamily::MaskedSoftmaxAttention);
  ASSERT_TRUE(
      sdpa_nomask_plan.stage_manifest.custom_kernel.external_buffer_abi.valid);
  EXPECT_EQ(
      sdpa_nomask_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>(
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
           GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::TensorOutput}));

  const auto sdpa_nomask_binding =
      make_msl_runtime_binding_plan_from_stage_manifest(
          sdpa_nomask_plan.stage_manifest);
  ASSERT_TRUE(sdpa_nomask_binding.valid());
  EXPECT_EQ(sdpa_nomask_binding.runtime_binding.inputs,
            std::vector<size_t>({0, 1, 2}));
  EXPECT_EQ(sdpa_nomask_binding.runtime_binding.input_arg_count, 4u);
  EXPECT_EQ(sdpa_nomask_binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3, 4}));

  const auto causal_plan = make_gfx_custom_kernel_stage_plan(
      "GfxSDPAWithCausalMask", "sdpa_causal_mask_kernel");
  ASSERT_TRUE(causal_plan.valid);
  ASSERT_TRUE(
      causal_plan.stage_manifest.custom_kernel.external_buffer_abi.valid);
  EXPECT_EQ(
      causal_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>(
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
           GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
           GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::TensorOutput}));
  const auto causal_binding = make_msl_runtime_binding_plan_from_stage_manifest(
      causal_plan.stage_manifest);
  ASSERT_TRUE(causal_binding.valid());
  EXPECT_EQ(causal_binding.runtime_binding.inputs,
            std::vector<size_t>({0, 1, 2, 3, 4}));
  EXPECT_EQ(causal_binding.runtime_binding.input_arg_count, 6u);
  EXPECT_EQ(causal_binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3, 4, 5, 6}));
}

TEST(GfxStagePolicyTest,
     MslRuntimeBindingPlanPreservesOutputBeforeRuntimeParamsOrder) {
  const auto gather_plan =
      make_gfx_custom_kernel_stage_plan("Gather", "gather_kernel");
  ASSERT_TRUE(gather_plan.valid);
  ASSERT_TRUE(
      gather_plan.stage_manifest.custom_kernel.external_buffer_abi.valid);
  EXPECT_EQ(
      gather_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams}));

  const auto gather_binding = make_msl_runtime_binding_plan_from_stage_manifest(
      gather_plan.stage_manifest);
  ASSERT_TRUE(gather_binding.valid());
  EXPECT_EQ(gather_binding.runtime_binding.inputs, std::vector<size_t>({0, 1}));
  EXPECT_EQ(gather_binding.runtime_binding.input_arg_count, 3u);
  EXPECT_EQ(gather_binding.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 1, 1}));
  EXPECT_EQ(gather_binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 3, 2}));

  const auto gather_elements_binding =
      make_msl_runtime_binding_plan_for_custom_kernel("GatherElements",
                                                      "gather_elements_kernel");
  ASSERT_TRUE(gather_elements_binding.valid());
  EXPECT_EQ(gather_elements_binding.runtime_binding.inputs,
            std::vector<size_t>({0, 1}));
  EXPECT_EQ(gather_elements_binding.runtime_binding.input_arg_count, 3u);
  EXPECT_EQ(gather_elements_binding.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 1, 1}));
  EXPECT_EQ(gather_elements_binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 3, 2}));

  const auto tile_binding = make_msl_runtime_binding_plan_for_custom_kernel(
      "Tile", "tile_kernel", {16, 4});
  ASSERT_TRUE(tile_binding.valid());
  EXPECT_EQ(tile_binding.runtime_binding.inputs, std::vector<size_t>({0}));
  EXPECT_EQ(tile_binding.runtime_binding.input_arg_count, 5u);
  EXPECT_EQ(tile_binding.scalar_arg_count, 2u);
  EXPECT_EQ(tile_binding.runtime_binding.scalar_args,
            std::vector<int32_t>({16, 4}));
  EXPECT_EQ(tile_binding.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 0, 0, 1, 1, 1, 1}));
  EXPECT_EQ(tile_binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 5, -1, -1, 1, 2, 3, 4}));

  const auto softmax_binding = make_msl_runtime_binding_plan_for_custom_kernel(
      "Softmax", "softmax_kernel");
  ASSERT_TRUE(softmax_binding.valid());
  EXPECT_EQ(softmax_binding.runtime_binding.inputs, std::vector<size_t>({0}));
  EXPECT_EQ(softmax_binding.runtime_binding.input_arg_count, 2u);
  EXPECT_EQ(softmax_binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 2, 1}));

  const auto split_binding =
      make_msl_runtime_binding_plan_for_direct_io_custom_kernel(
          "VariadicSplit", "split_kernel", 1, 3);
  ASSERT_TRUE(split_binding.valid());
  EXPECT_EQ(split_binding.runtime_binding.inputs, std::vector<size_t>({0}));
  EXPECT_EQ(split_binding.runtime_binding.input_arg_count, 1u);
  EXPECT_EQ(split_binding.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 1, 1}));
  EXPECT_EQ(split_binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3}));
  EXPECT_TRUE(split_binding.stage_manifest.custom_kernel.external_buffer_abi
                  .roles.empty());
  EXPECT_EQ(split_binding.stage_manifest.custom_kernel.external_buffer_abi
                .direct_input_count,
            1u);
  EXPECT_EQ(split_binding.stage_manifest.custom_kernel.external_buffer_abi
                .direct_output_count,
            3u);
  EXPECT_EQ(
      materialize_gfx_kernel_external_buffer_roles(
          split_binding.stage_manifest.custom_kernel.external_buffer_abi),
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::TensorOutput}));
  const auto split_source_plan = make_direct_split_msl_kernel_source_plan(
      "VariadicSplit", ov::element::f32, ov::Shape{1, 6, 2}, {2, 2, 2}, 6, 2);
  ASSERT_TRUE(split_source_plan.valid());
  EXPECT_EQ(split_source_plan.source.entry_point, "split_kernel");
  EXPECT_EQ(split_source_plan.source.signature.arg_count, 4u);
  EXPECT_EQ(split_source_plan.source.signature.output_arg_count, 3u);
  EXPECT_EQ(split_source_plan.binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3}));

  const auto select_binding = make_msl_runtime_binding_plan_for_custom_kernel(
      "Select", "select_kernel", {16, 4});
  ASSERT_TRUE(select_binding.valid());
  EXPECT_EQ(select_binding.runtime_binding.inputs,
            std::vector<size_t>({0, 1, 2}));
  EXPECT_EQ(select_binding.runtime_binding.input_arg_count, 7u);
  EXPECT_EQ(select_binding.scalar_arg_count, 2u);
  EXPECT_EQ(select_binding.runtime_binding.scalar_args,
            std::vector<int32_t>({16, 4}));
  EXPECT_EQ(select_binding.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 1, 1, 0, 0, 1, 1, 1, 1}));
  EXPECT_EQ(select_binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 7, -1, -1, 3, 4, 5, 6}));

  const auto rms_binding =
      make_msl_runtime_binding_plan_for_custom_kernel("RMS", "rms_kernel");
  ASSERT_TRUE(rms_binding.valid());
  EXPECT_EQ(rms_binding.runtime_binding.inputs, std::vector<size_t>({0, 1}));
  EXPECT_EQ(rms_binding.runtime_binding.input_arg_count, 2u);
  EXPECT_EQ(rms_binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2}));

  const auto rms_residual_binding =
      make_msl_runtime_binding_plan_for_custom_kernel("RMSResidual",
                                                      "rms_kernel");
  ASSERT_TRUE(rms_residual_binding.valid());
  EXPECT_EQ(rms_residual_binding.runtime_binding.inputs,
            std::vector<size_t>({0, 1, 2}));
  EXPECT_EQ(rms_residual_binding.runtime_binding.input_arg_count, 3u);
  EXPECT_EQ(rms_residual_binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3}));

  const auto rope_binding =
      make_msl_runtime_binding_plan_for_custom_kernel("RoPE", "rope_kernel");
  ASSERT_TRUE(rope_binding.valid());
  EXPECT_EQ(rope_binding.runtime_binding.inputs,
            std::vector<size_t>({0, 1, 2}));
  EXPECT_EQ(rope_binding.runtime_binding.input_arg_count, 3u);
  EXPECT_EQ(rope_binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3}));

  const auto rope_pos_binding = make_msl_runtime_binding_plan_for_custom_kernel(
      "RoPEWithPosition", "rope_kernel");
  ASSERT_TRUE(rope_pos_binding.valid());
  EXPECT_EQ(rope_pos_binding.runtime_binding.inputs,
            std::vector<size_t>({0, 1, 2, 3}));
  EXPECT_EQ(rope_pos_binding.runtime_binding.input_arg_count, 4u);
  EXPECT_EQ(rope_pos_binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3, 4}));
}

TEST(GfxStagePolicyTest,
     CustomKernelStagePlanCoversExistingUnaryAndBinaryMslOps) {
  const auto elu_plan =
      make_gfx_custom_kernel_stage_plan("Elu", "unary_kernel");
  ASSERT_TRUE(elu_plan.valid);
  EXPECT_EQ(elu_plan.family, GfxKernelFamily::EltwiseFusedBuffer);
  EXPECT_EQ(elu_plan.stage_manifest.custom_kernel.entry_point,
            "eltwise_fused_buffer");
  EXPECT_EQ(elu_plan.stage_manifest.custom_kernel.kernel_family_id,
            static_cast<uint32_t>(GfxKernelFamily::EltwiseFusedBuffer));
  ASSERT_TRUE(elu_plan.stage_manifest.custom_kernel.external_buffer_abi.valid);
  EXPECT_EQ(
      elu_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::ScalarParam}));

  const auto sqdiff_plan =
      make_gfx_custom_kernel_stage_plan("SquaredDifference", "eltwise_kernel");
  ASSERT_TRUE(sqdiff_plan.valid);
  EXPECT_EQ(sqdiff_plan.family, GfxKernelFamily::EltwiseFusedBuffer);
  EXPECT_EQ(sqdiff_plan.stage_manifest.custom_kernel.entry_point,
            "eltwise_fused_buffer");
  ASSERT_TRUE(
      sqdiff_plan.stage_manifest.custom_kernel.external_buffer_abi.valid);
  EXPECT_EQ(
      sqdiff_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>(
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
           GfxKernelBufferRole::TensorOutput, GfxKernelBufferRole::ScalarParam,
           GfxKernelBufferRole::ScalarParam, GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams}));

  const auto reduce_plan =
      make_gfx_custom_kernel_stage_plan("ReduceMean", "reduce_kernel");
  ASSERT_TRUE(reduce_plan.valid);
  EXPECT_EQ(reduce_plan.family, GfxKernelFamily::ReductionBuffer);
  EXPECT_EQ(reduce_plan.stage_manifest.custom_kernel.entry_point,
            "reduction_buffer");
  ASSERT_TRUE(reduce_plan.stage_manifest.custom_kernel.dispatch_policy.valid);
  EXPECT_EQ(reduce_plan.stage_manifest.custom_kernel.dispatch_policy.grid,
            GfxKernelDispatchGrid::Linear1D);
  EXPECT_EQ(reduce_plan.stage_manifest.custom_kernel.dispatch_policy
                .threads_per_threadgroup,
            128u);
  ASSERT_TRUE(
      reduce_plan.stage_manifest.custom_kernel.external_buffer_abi.valid);
  ASSERT_EQ(
      reduce_plan.stage_manifest.custom_kernel.external_buffer_abi.roles.size(),
      9u);
  EXPECT_EQ(
      reduce_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[0],
      GfxKernelBufferRole::TensorInput);
  EXPECT_EQ(
      reduce_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[1],
      GfxKernelBufferRole::TensorOutput);
  EXPECT_EQ(
      reduce_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[2],
      GfxKernelBufferRole::ScalarParam);
  EXPECT_EQ(
      reduce_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[3],
      GfxKernelBufferRole::ScalarParam);
  for (size_t i = 4; i < reduce_plan.stage_manifest.custom_kernel
                             .external_buffer_abi.roles.size();
       ++i) {
    EXPECT_EQ(
        reduce_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[i],
        GfxKernelBufferRole::RuntimeParams);
  }

  const auto softmax_plan =
      make_gfx_custom_kernel_stage_plan("Softmax", "softmax_kernel");
  ASSERT_TRUE(softmax_plan.valid);
  EXPECT_EQ(softmax_plan.family, GfxKernelFamily::MaskedSoftmaxAttention);
  ASSERT_TRUE(softmax_plan.stage_manifest.valid);
  EXPECT_EQ(softmax_plan.stage_manifest.stage_family,
            GfxKernelStageFamily::AttentionSoftmax);
  EXPECT_EQ(softmax_plan.stage_manifest.backend_domain,
            GfxKernelBackendDomain::AppleMsl);
  EXPECT_EQ(softmax_plan.stage_manifest.execution_kind,
            GfxKernelExecutionKind::CustomKernel);
  EXPECT_EQ(softmax_plan.stage_manifest.storage, GfxKernelStorageKind::Buffer);
  ASSERT_TRUE(softmax_plan.stage_manifest.custom_kernel.valid);
  EXPECT_EQ(softmax_plan.stage_manifest.custom_kernel.kernel_family,
            "masked_softmax_attention");
  ASSERT_TRUE(softmax_plan.stage_manifest.custom_kernel.dispatch_policy.valid);
  EXPECT_EQ(softmax_plan.stage_manifest.custom_kernel.dispatch_policy.grid,
            GfxKernelDispatchGrid::Linear1D);
  EXPECT_EQ(softmax_plan.stage_manifest.custom_kernel.dispatch_policy
                .threads_per_threadgroup,
            128u);
  ASSERT_TRUE(
      softmax_plan.stage_manifest.custom_kernel.external_buffer_abi.valid);
  EXPECT_EQ(
      softmax_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams}));

  const auto topk_plan =
      make_gfx_custom_kernel_stage_plan("TopK", "topk_kernel");
  ASSERT_TRUE(topk_plan.valid);
  EXPECT_EQ(topk_plan.family, GfxKernelFamily::GatherScatterIndexed);
  ASSERT_TRUE(topk_plan.stage_manifest.custom_kernel.external_buffer_abi.valid);
  EXPECT_EQ(
      topk_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::TensorOutput}));

  const auto gather_plan =
      make_gfx_custom_kernel_stage_plan("Gather", "gather_kernel");
  ASSERT_TRUE(gather_plan.valid);
  EXPECT_EQ(gather_plan.family, GfxKernelFamily::GatherScatterIndexed);
  ASSERT_TRUE(
      gather_plan.stage_manifest.custom_kernel.external_buffer_abi.valid);
  EXPECT_EQ(
      gather_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams}));

  const auto slice_plan =
      make_gfx_custom_kernel_stage_plan("Slice", "slice_kernel");
  ASSERT_TRUE(slice_plan.valid);
  EXPECT_EQ(slice_plan.family, GfxKernelFamily::GatherScatterIndexed);
  ASSERT_TRUE(
      slice_plan.stage_manifest.custom_kernel.external_buffer_abi.valid);
  ASSERT_EQ(
      slice_plan.stage_manifest.custom_kernel.external_buffer_abi.roles.size(),
      8u);
  EXPECT_EQ(
      slice_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[0],
      GfxKernelBufferRole::TensorInput);
  EXPECT_EQ(
      slice_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[1],
      GfxKernelBufferRole::TensorOutput);
  for (size_t i = 2;
       i <
       slice_plan.stage_manifest.custom_kernel.external_buffer_abi.roles.size();
       ++i) {
    EXPECT_EQ(
        slice_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[i],
        GfxKernelBufferRole::RuntimeParams);
  }

  const auto concat_plan =
      make_gfx_custom_kernel_stage_plan("Concat", "concat_kernel");
  ASSERT_TRUE(concat_plan.valid);
  EXPECT_EQ(concat_plan.family, GfxKernelFamily::ConcatSplitGeneric);
  EXPECT_EQ(
      concat_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams}));

  const auto concat_binary_plan =
      make_gfx_custom_kernel_stage_plan("Concat", "concat_binary_kernel");
  ASSERT_TRUE(concat_binary_plan.valid);
  EXPECT_EQ(
      concat_binary_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams}));

  const auto transpose_plan =
      make_gfx_custom_kernel_stage_plan("Transpose", "transpose_kernel");
  ASSERT_TRUE(transpose_plan.valid);
  EXPECT_EQ(transpose_plan.family, GfxKernelFamily::TransposePackND);
  EXPECT_EQ(
      transpose_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams}));

  const auto tile_plan =
      make_gfx_custom_kernel_stage_plan("Tile", "tile_kernel");
  ASSERT_TRUE(tile_plan.valid);
  EXPECT_EQ(tile_plan.family, GfxKernelFamily::GatherScatterIndexed);
  ASSERT_EQ(
      tile_plan.stage_manifest.custom_kernel.external_buffer_abi.roles.size(),
      8u);
  EXPECT_EQ(tile_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[0],
            GfxKernelBufferRole::TensorInput);
  EXPECT_EQ(tile_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[1],
            GfxKernelBufferRole::TensorOutput);
  EXPECT_EQ(tile_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[2],
            GfxKernelBufferRole::ScalarParam);
  EXPECT_EQ(tile_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[3],
            GfxKernelBufferRole::ScalarParam);
  for (size_t i = 4;
       i <
       tile_plan.stage_manifest.custom_kernel.external_buffer_abi.roles.size();
       ++i) {
    EXPECT_EQ(
        tile_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[i],
        GfxKernelBufferRole::RuntimeParams);
  }

  const auto broadcast_plan =
      make_gfx_custom_kernel_stage_plan("Broadcast", "broadcast_kernel");
  ASSERT_TRUE(broadcast_plan.valid);
  EXPECT_EQ(broadcast_plan.family, GfxKernelFamily::EltwiseFusedBuffer);
  ASSERT_EQ(broadcast_plan.stage_manifest.custom_kernel.external_buffer_abi
                .roles.size(),
            9u);
  EXPECT_EQ(
      broadcast_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[0],
      GfxKernelBufferRole::TensorInput);
  EXPECT_EQ(
      broadcast_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[1],
      GfxKernelBufferRole::TensorOutput);
  EXPECT_EQ(
      broadcast_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[2],
      GfxKernelBufferRole::ScalarParam);
  EXPECT_EQ(
      broadcast_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[3],
      GfxKernelBufferRole::ScalarParam);
  EXPECT_EQ(
      broadcast_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[4],
      GfxKernelBufferRole::ScalarParam);

  const auto select_plan =
      make_gfx_custom_kernel_stage_plan("Select", "select_kernel");
  ASSERT_TRUE(select_plan.valid);
  EXPECT_EQ(select_plan.family, GfxKernelFamily::EltwiseFusedBuffer);
  EXPECT_EQ(
      select_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[0],
      GfxKernelBufferRole::TensorInput);
  EXPECT_EQ(
      select_plan.stage_manifest.custom_kernel.external_buffer_abi.roles[3],
      GfxKernelBufferRole::TensorOutput);
  ASSERT_EQ(
      select_plan.stage_manifest.custom_kernel.external_buffer_abi.roles.size(),
      10u);

  const auto range_plan =
      make_gfx_custom_kernel_stage_plan("Range", "range_kernel");
  ASSERT_TRUE(range_plan.valid);
  EXPECT_EQ(range_plan.family, GfxKernelFamily::GatherScatterIndexed);
  EXPECT_EQ(
      range_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>(
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
           GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
           GfxKernelBufferRole::ScalarParam}));
  const auto range_binding = make_kernel_runtime_binding_plan_for_custom_kernel(
      "Range", "range_kernel", {32}, GfxKernelBackendDomain::Spirv,
      GfxKernelStorageKind::Buffer, "spirv:buffer:");
  ASSERT_TRUE(range_binding.valid);
  EXPECT_EQ(range_binding.stage_manifest.backend_domain,
            GfxKernelBackendDomain::Spirv);
  EXPECT_EQ(range_binding.runtime_binding.inputs,
            std::vector<size_t>({0, 1, 2}));
  EXPECT_EQ(range_binding.runtime_binding.input_arg_count, 3u);
  EXPECT_EQ(range_binding.runtime_binding.scalar_args,
            std::vector<int32_t>({32}));
  EXPECT_EQ(range_binding.runtime_binding.operand_kinds,
            std::vector<int32_t>({1, 1, 1, 1, 0}));
  EXPECT_EQ(range_binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 2, 3, -1}));

  const auto scatter_init_plan = make_gfx_custom_kernel_stage_plan(
      "ScatterElementsUpdate", "scatter_elements_init");
  ASSERT_TRUE(scatter_init_plan.valid);
  EXPECT_EQ(
      scatter_init_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams}));

  const auto scatter_elements_update_plan = make_gfx_custom_kernel_stage_plan(
      "ScatterElementsUpdate", "scatter_elements_update");
  ASSERT_TRUE(scatter_elements_update_plan.valid);
  EXPECT_EQ(
      scatter_elements_update_plan.stage_manifest.custom_kernel
          .external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams}));

  const auto scatter_update_plan = make_gfx_custom_kernel_stage_plan(
      "ScatterUpdate", "scatter_update_kernel");
  ASSERT_TRUE(scatter_update_plan.valid);
  EXPECT_EQ(
      scatter_update_plan.stage_manifest.custom_kernel.external_buffer_abi
          .roles,
      std::vector<GfxKernelBufferRole>(
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
           GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorOutput,
           GfxKernelBufferRole::RuntimeParams}));

  const auto pool_plan =
      make_gfx_custom_kernel_stage_plan("MaxPool", "pool2d_kernel");
  ASSERT_TRUE(pool_plan.valid);
  EXPECT_EQ(pool_plan.family, GfxKernelFamily::Pool2DWindow);
  EXPECT_EQ(pool_plan.stage_manifest.stage_family,
            GfxKernelStageFamily::Pooling);
  EXPECT_EQ(
      pool_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::RuntimeParams,
                                        GfxKernelBufferRole::TensorOutput}));

  const auto batchnorm_plan = make_gfx_custom_kernel_stage_plan(
      "BatchNormInference", "batchnorm2d_kernel");
  ASSERT_TRUE(batchnorm_plan.valid);
  EXPECT_EQ(batchnorm_plan.family, GfxKernelFamily::BatchNormBuffer);
  EXPECT_EQ(batchnorm_plan.stage_manifest.stage_family,
            GfxKernelStageFamily::Eltwise);
  EXPECT_EQ(
      batchnorm_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::ConstTensor,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams}));

  const auto conv3d_plan =
      make_gfx_custom_kernel_stage_plan("Convolution", "conv3d_kernel");
  ASSERT_TRUE(conv3d_plan.valid);
  EXPECT_EQ(conv3d_plan.family, GfxKernelFamily::Conv3DDirectOrIm2col);
  EXPECT_EQ(conv3d_plan.stage_manifest.custom_kernel.entry_point,
            "conv3d_direct_or_im2col");
  EXPECT_EQ(
      conv3d_plan.stage_manifest.custom_kernel.external_buffer_abi.roles,
      std::vector<GfxKernelBufferRole>({GfxKernelBufferRole::TensorInput,
                                        GfxKernelBufferRole::ConstTensor,
                                        GfxKernelBufferRole::TensorOutput,
                                        GfxKernelBufferRole::RuntimeParams}));
  const auto conv3d_binding = make_msl_runtime_binding_plan_from_stage_manifest(
      conv3d_plan.stage_manifest);
  ASSERT_TRUE(conv3d_binding.valid());
  EXPECT_EQ(conv3d_binding.runtime_binding.inputs, std::vector<size_t>({0}));
  EXPECT_EQ(conv3d_binding.runtime_binding.input_arg_count, 3u);
  EXPECT_EQ(conv3d_binding.runtime_binding.operand_arg_indices,
            std::vector<int32_t>({0, 1, 3, 2}));
  ASSERT_TRUE(conv3d_plan.stage_manifest.custom_kernel.dispatch_policy.valid);
  EXPECT_EQ(conv3d_plan.stage_manifest.custom_kernel.dispatch_policy.grid,
            GfxKernelDispatchGrid::Linear1D);
  EXPECT_EQ(conv3d_plan.stage_manifest.custom_kernel.dispatch_policy
                .threads_per_threadgroup,
            128u);
}

TEST(GfxStagePolicyTest, MpsrtBuilderPlanSerializesMslDispatchStage) {
  const auto add = make_large_add_node();
  const auto plan = select_stage_optimization_plan(nullptr, GpuBackend::Metal,
                                                   "Add", add, ov::element::f16,
                                                   /*has_bias=*/false,
                                                   /*has_activation=*/false,
                                                   /*has_batchnorm=*/false, {});
  const auto stage = gfx_mpsrt_make_stage_desc(plan, "Add");
  const auto lhs = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagExternalIo);
  const auto rhs = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagExternalIo);
  const auto out = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagTransient);
  const auto builder_plan = gfx_mpsrt_make_builder_plan(
      stage, {lhs, rhs}, {out}, gfx_mpsrt_stage_record_key(stage));

  ASSERT_TRUE(builder_plan.valid);
  ASSERT_EQ(builder_plan.records.size(), 5u);
  EXPECT_EQ(builder_plan.records[0].symbol, "ovgfx_mpsrt_model_begin");
  EXPECT_EQ(builder_plan.records[1].symbol, "ovgfx_mpsrt_add_tensor");
  EXPECT_EQ(builder_plan.records[2].symbol, "ovgfx_mpsrt_add_tensor");
  EXPECT_EQ(builder_plan.records[3].kind,
            GfxMpsrtBuilderRecordKind::EncodeStage);
  EXPECT_EQ(builder_plan.records[3].symbol, "ovgfx_mpsrt_encode_dispatch");
  EXPECT_EQ(builder_plan.records[3].kernel_name, "eltwise_fused_buffer");
  EXPECT_EQ(builder_plan.records[3].dispatch_kernel_family,
            "eltwise_fused_buffer");
  EXPECT_EQ(builder_plan.records[3].dispatch_entry_point,
            "eltwise_fused_buffer");
  EXPECT_EQ(builder_plan.records[3].dispatch_kernel_family_id,
            static_cast<uint32_t>(GfxKernelFamily::EltwiseFusedBuffer));
  EXPECT_EQ(builder_plan.records[3].dispatch_flags,
            GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
  EXPECT_EQ(builder_plan.records[3].dispatch_threads_per_threadgroup, 256u);
  EXPECT_TRUE(builder_plan.records[3].dispatch_precompiled_kernel_required);
  EXPECT_EQ(builder_plan.records[3].msl_dispatch_desc.kernel_family,
            static_cast<uint32_t>(GfxKernelFamily::EltwiseFusedBuffer));
  EXPECT_EQ(builder_plan.records[3].msl_dispatch_desc.storage,
            static_cast<uint32_t>(GfxMpsrtStorage::Buffer));
  EXPECT_EQ(builder_plan.records[3].msl_dispatch_desc.layout,
            static_cast<uint32_t>(GfxMpsrtLayout::Linear));
  EXPECT_EQ(builder_plan.records[3].msl_dispatch_desc.threads_per_threadgroup,
            256u);
  EXPECT_EQ(builder_plan.records[3].msl_dispatch_desc.input_count, 2u);
  EXPECT_EQ(builder_plan.records[3].msl_dispatch_desc.output_count, 1u);
  EXPECT_EQ(builder_plan.records[3].msl_dispatch_desc.flags,
            GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
  EXPECT_EQ(builder_plan.records[3].inputs,
            std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_EQ(builder_plan.records[3].outputs, std::vector<GfxMpsrtValue>({2u}));
  EXPECT_EQ(builder_plan.records[3].kernel_buffer_order,
            std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
  EXPECT_EQ(builder_plan.records[3].tensor_descs[0].byte_length,
            1u * 64u * 80u * 80u * 2u);
  EXPECT_EQ(builder_plan.records[4].symbol, "ovgfx_mpsrt_model_end");
}

TEST(GfxStagePolicyTest,
     MpsrtBuilderPlanUsesManifestRolesForMslKernelBufferOrder) {
  const auto add = make_large_add_node();
  const auto plan = select_stage_optimization_plan(nullptr, GpuBackend::Metal,
                                                   "Add", add, ov::element::f16,
                                                   /*has_bias=*/false,
                                                   /*has_activation=*/false,
                                                   /*has_batchnorm=*/false, {});
  auto stage = gfx_mpsrt_make_stage_desc(plan, "Add");
  stage.stage_manifest.custom_kernel.external_buffer_abi =
      make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorOutput,
                                 GfxKernelBufferRole::TensorInput,
                                 GfxKernelBufferRole::TensorInput});
  const auto lhs = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagExternalIo);
  const auto rhs = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagExternalIo);
  const auto out = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagTransient);
  const auto builder_plan = gfx_mpsrt_make_builder_plan(
      stage, {lhs, rhs}, {out}, gfx_mpsrt_stage_record_key(stage));

  ASSERT_TRUE(builder_plan.valid);
  ASSERT_EQ(builder_plan.records.size(), 5u);
  ASSERT_EQ(builder_plan.records[3].kind,
            GfxMpsrtBuilderRecordKind::EncodeStage);
  EXPECT_EQ(builder_plan.records[3].inputs,
            std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_EQ(builder_plan.records[3].outputs, std::vector<GfxMpsrtValue>({2u}));
  EXPECT_EQ(builder_plan.records[3].kernel_buffer_order,
            std::vector<GfxMpsrtValue>({2u, 0u, 1u}));
}

TEST(GfxStagePolicyTest,
     MpsrtManifestAdapterMaterializesExternalKernelBufferOrder) {
  const std::vector<GfxMpsrtExternalBufferRole> roles = {
      GfxMpsrtExternalBufferRole::TensorInput,
      GfxMpsrtExternalBufferRole::TensorOutput,
      GfxMpsrtExternalBufferRole::RuntimeParams};
  const std::vector<GfxMpsrtValue> external_values = {0u, 1u, 2u};

  EXPECT_EQ(gfx_mpsrt_kernel_buffer_order_from_external_values(roles,
                                                               external_values),
            external_values);
  EXPECT_TRUE(
      gfx_mpsrt_kernel_buffer_order_from_kernel_abi({}, {0u}, {1u}).empty());
  EXPECT_EQ(gfx_mpsrt_kernel_buffer_order_from_kernel_abi(
                make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorOutput,
                                           GfxKernelBufferRole::TensorInput,
                                           GfxKernelBufferRole::TensorInput}),
                {0u, 1u}, {2u}),
            std::vector<GfxMpsrtValue>({2u, 0u, 1u}));
  EXPECT_EQ(gfx_mpsrt_kernel_buffer_order_from_kernel_abi(
                make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                           GfxKernelBufferRole::TensorInput,
                                           GfxKernelBufferRole::TensorOutput}),
                {0u}, {1u}),
            std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_TRUE(gfx_mpsrt_kernel_buffer_order_from_external_values(
                  {GfxMpsrtExternalBufferRole::TensorInput}, external_values)
                  .empty());
  EXPECT_TRUE(gfx_mpsrt_kernel_buffer_order_from_external_values(
                  {GfxMpsrtExternalBufferRole::Unknown}, {0u})
                  .empty());
}

TEST(GfxStagePolicyTest,
     MpsrtMultiStageBuilderPlanSerializesMpsGemmPlusMslDispatch) {
  GfxMpsrtStageDesc gemm_stage{};
  gemm_stage.kind = GfxMpsrtStageKind::MPSGemm;
  gemm_stage.domain = GfxStageBackendDomain::AppleMps;
  gemm_stage.input_storage = GfxMpsrtStorage::Matrix;
  gemm_stage.output_storage = GfxMpsrtStorage::Matrix;
  gemm_stage.layout = GfxMpsrtLayout::RowMajor;
  gemm_stage.uses_vendor_primitive = true;
  gemm_stage.stage_type = "MatMul";
  gemm_stage.kernel_name = "mps_gemm";
  gemm_stage.builder_symbol = gfx_mpsrt_builder_symbol(gemm_stage.kind);
  gemm_stage.specialization_key = "apple_mps:matrix:MatMul";
  gemm_stage.gemm_desc.alpha = 1.0f;
  gemm_stage.stage_manifest = make_gfx_vendor_stage_manifest(
      GfxKernelStageFamily::Gemm, GfxKernelBackendDomain::AppleMps,
      GfxKernelStorageKind::Matrix, gemm_stage.specialization_key);

  GfxMpsrtStageDesc epilogue_stage{};
  epilogue_stage.kind = GfxMpsrtStageKind::MSLDispatch;
  epilogue_stage.domain = GfxStageBackendDomain::AppleMsl;
  epilogue_stage.input_storage = GfxMpsrtStorage::Buffer;
  epilogue_stage.output_storage = GfxMpsrtStorage::Buffer;
  epilogue_stage.layout = GfxMpsrtLayout::Linear;
  epilogue_stage.uses_custom_kernel = true;
  epilogue_stage.stage_type = "MatMulEpilogue";
  epilogue_stage.kernel_name = "eltwise_fused_buffer";
  epilogue_stage.builder_symbol = gfx_mpsrt_builder_symbol(epilogue_stage.kind);
  epilogue_stage.specialization_key = "apple_msl:buffer:MatMulEpilogue";
  epilogue_stage.stage_manifest = make_gfx_custom_kernel_stage_manifest(
      GfxKernelStageFamily::Eltwise, GfxKernelBackendDomain::AppleMsl,
      GfxKernelStorageKind::Buffer, epilogue_stage.specialization_key,
      make_gfx_custom_kernel_manifest(
          "eltwise_fused_buffer",
          static_cast<uint32_t>(GfxKernelFamily::EltwiseFusedBuffer),
          "eltwise_fused_buffer",
          make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                     GfxKernelBufferRole::TensorOutput}),
          make_gfx_kernel_linear_dispatch_policy(
              256,
              /*precompiled_binary_required=*/true)));

  const auto lhs = gfx_mpsrt_make_tensor_desc({1, 128, 256}, ov::element::f16,
                                              GfxStageStorageKind::Matrix,
                                              GfxMpsrtTensorFlagExternalIo);
  const auto rhs = gfx_mpsrt_make_tensor_desc({1, 256, 64}, ov::element::f16,
                                              GfxStageStorageKind::Matrix,
                                              GfxMpsrtTensorFlagExternalIo);
  const auto gemm = gfx_mpsrt_make_tensor_desc({1, 128, 64}, ov::element::f16,
                                               GfxStageStorageKind::Matrix);
  const auto out = gfx_mpsrt_make_tensor_desc({1, 128, 64}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagExternalIo);

  GfxMpsrtProgram program{};
  program.record_key = "mps_gemm_plus_msl_epilogue_model|MatMul";
  program.multi_stage = true;
  program.inputs = {lhs, rhs};
  program.output_values = {3u};
  program.stages.push_back({gemm_stage,
                            gfx_mpsrt_stage_record_key(gemm_stage),
                            {0u, 1u},
                            {2u},
                            {gemm}});
  program.stages.push_back({epilogue_stage,
                            gfx_mpsrt_stage_record_key(epilogue_stage),
                            {2u},
                            {3u},
                            {out}});
  program.external_buffer_abi.valid = true;
  program.external_buffer_abi.has_buffer_count = true;
  program.external_buffer_abi.has_output_buffer_count = true;
  program.external_buffer_abi.has_buffer_roles = true;
  program.external_buffer_abi.buffer_count = 3u;
  program.external_buffer_abi.output_buffer_count = 1u;
  program.external_buffer_abi.buffer_roles = {
      GfxMpsrtExternalBufferRole::TensorInput,
      GfxMpsrtExternalBufferRole::TensorInput,
      GfxMpsrtExternalBufferRole::TensorOutput};
  program.valid = gfx_mpsrt_validate_program(program, nullptr);

  const auto validation = gfx_mpsrt_validate_program(program);
  ASSERT_TRUE(validation.valid) << validation.error;

  GfxMpsrtBuilderPlan program_builder_plan{};
  ASSERT_TRUE(
      gfx_mpsrt_build_builder_plan_from_program(program, program_builder_plan));
  ASSERT_TRUE(program_builder_plan.valid);
  EXPECT_EQ(program_builder_plan.records.size(), 6u);
  EXPECT_TRUE(program_builder_plan.external_buffer_abi_valid);
  EXPECT_EQ(program_builder_plan.external_buffer_roles,
            program.external_buffer_abi.buffer_roles);

  auto invalid_program = program;
  invalid_program.stages[1].inputs = {42u};
  std::string validation_error;
  EXPECT_FALSE(gfx_mpsrt_validate_program(invalid_program, &validation_error));
  EXPECT_NE(validation_error.find("reads a value before it is materialized"),
            std::string::npos);

  auto builder_plan = gfx_mpsrt_make_multi_stage_builder_plan(
      "mps_gemm_plus_msl_epilogue_model|MatMul", {lhs, rhs},
      {GfxMpsrtBuilderStageSpec{gemm_stage,
                                gfx_mpsrt_stage_record_key(gemm_stage),
                                {0u, 1u},
                                {2u},
                                {gemm}},
       GfxMpsrtBuilderStageSpec{epilogue_stage,
                                gfx_mpsrt_stage_record_key(epilogue_stage),
                                {2u},
                                {3u},
                                {out}}},
      {3u});
  builder_plan.external_buffer_roles = {
      GfxMpsrtExternalBufferRole::TensorInput,
      GfxMpsrtExternalBufferRole::TensorInput,
      GfxMpsrtExternalBufferRole::TensorOutput};

  ASSERT_TRUE(builder_plan.valid);
  ASSERT_EQ(builder_plan.records.size(), 6u);
  EXPECT_EQ(builder_plan.input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_EQ(builder_plan.output_values, std::vector<GfxMpsrtValue>({3u}));
  ASSERT_EQ(builder_plan.records[3].kind,
            GfxMpsrtBuilderRecordKind::EncodeStage);
  EXPECT_EQ(builder_plan.records[3].stage_kind, GfxMpsrtStageKind::MPSGemm);
  EXPECT_EQ(builder_plan.records[3].inputs,
            std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_EQ(builder_plan.records[3].outputs, std::vector<GfxMpsrtValue>({2u}));
  ASSERT_EQ(builder_plan.records[4].kind,
            GfxMpsrtBuilderRecordKind::EncodeStage);
  EXPECT_EQ(builder_plan.records[4].stage_kind, GfxMpsrtStageKind::MSLDispatch);
  EXPECT_EQ(builder_plan.records[4].inputs, std::vector<GfxMpsrtValue>({2u}));
  EXPECT_EQ(builder_plan.records[4].outputs, std::vector<GfxMpsrtValue>({3u}));
  EXPECT_EQ(builder_plan.records[4].kernel_buffer_order,
            std::vector<GfxMpsrtValue>({2u, 3u}));

  runtime_mpsrt::MpsrtModel model;
  std::string error;
  ASSERT_TRUE(runtime_mpsrt::build_mpsrt_model_from_builder_plan(
      builder_plan, model, &error))
      << error;
  ASSERT_EQ(model.stages.size(), 2u);
  EXPECT_EQ(model.stages[0].kind, GfxMpsrtStageKind::MPSGemm);
  EXPECT_EQ(model.stages[1].kind, GfxMpsrtStageKind::MSLDispatch);
  EXPECT_EQ(model.semantic_input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_EQ(model.semantic_output_values, std::vector<GfxMpsrtValue>({3u}));
  EXPECT_EQ(model.external_values, std::vector<GfxMpsrtValue>({0u, 1u, 3u}));
  EXPECT_EQ(model.external_buffer_roles,
            std::vector<GfxMpsrtExternalBufferRole>(
                {GfxMpsrtExternalBufferRole::TensorInput,
                 GfxMpsrtExternalBufferRole::TensorInput,
                 GfxMpsrtExternalBufferRole::TensorOutput}));

  ASSERT_TRUE(
      runtime_mpsrt::adapt_mpsrt_model_to_external_buffer_abi(
          model, 3u, 1u, &error))
      << error;
  EXPECT_EQ(model.external_values, std::vector<GfxMpsrtValue>({0u, 1u, 3u}));
  EXPECT_EQ(model.external_input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_EQ(model.external_output_values, std::vector<GfxMpsrtValue>({3u}));

  auto output_first_builder_plan = builder_plan;
  output_first_builder_plan.external_buffer_roles = {
      GfxMpsrtExternalBufferRole::TensorOutput,
      GfxMpsrtExternalBufferRole::TensorInput,
      GfxMpsrtExternalBufferRole::TensorInput};
  runtime_mpsrt::MpsrtModel output_first_model;
  ASSERT_TRUE(runtime_mpsrt::build_mpsrt_model_from_builder_plan(
      output_first_builder_plan, output_first_model, &error))
      << error;
  ASSERT_TRUE(
      runtime_mpsrt::adapt_mpsrt_model_to_external_buffer_abi(
          output_first_model, 3u, 1u, &error))
      << error;
  EXPECT_EQ(output_first_model.semantic_input_values,
            std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_EQ(output_first_model.semantic_output_values,
            std::vector<GfxMpsrtValue>({3u}));
  EXPECT_EQ(output_first_model.external_values,
            std::vector<GfxMpsrtValue>({3u, 0u, 1u}));
  EXPECT_EQ(output_first_model.external_input_values,
            std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_EQ(output_first_model.external_output_values,
            std::vector<GfxMpsrtValue>({3u}));

  auto runtime_param_builder_plan = builder_plan;
  runtime_param_builder_plan.external_buffer_roles = {
      GfxMpsrtExternalBufferRole::TensorInput,
      GfxMpsrtExternalBufferRole::TensorInput,
      GfxMpsrtExternalBufferRole::RuntimeParams,
      GfxMpsrtExternalBufferRole::TensorOutput};
  runtime_mpsrt::MpsrtModel runtime_param_model;
  ASSERT_TRUE(runtime_mpsrt::build_mpsrt_model_from_builder_plan(
      runtime_param_builder_plan, runtime_param_model, &error))
      << error;
  ASSERT_TRUE(
      runtime_mpsrt::adapt_mpsrt_model_to_external_buffer_abi(
          runtime_param_model, 4u, 1u, &error))
      << error;
  EXPECT_EQ(runtime_param_model.external_values,
            std::vector<GfxMpsrtValue>({0u, 1u, 3u}));
  EXPECT_EQ(runtime_param_model.external_buffer_roles,
            std::vector<GfxMpsrtExternalBufferRole>(
                {GfxMpsrtExternalBufferRole::TensorInput,
                 GfxMpsrtExternalBufferRole::TensorInput,
                 GfxMpsrtExternalBufferRole::RuntimeParams,
                 GfxMpsrtExternalBufferRole::TensorOutput}));
  ASSERT_EQ(runtime_param_model.external_buffer_bindings.size(), 4u);
  ASSERT_EQ(runtime_param_model.resources.size(), 5u);
  EXPECT_EQ(runtime_param_model.external_buffer_bindings[0].arg_index, 0u);
  EXPECT_EQ(runtime_param_model.external_buffer_bindings[0].resource_index, 0u);
  EXPECT_EQ(runtime_param_model.resources[0].role,
            GfxMpsrtExternalBufferRole::TensorInput);
  EXPECT_TRUE(runtime_param_model.resources[0].has_tensor_value);
  EXPECT_EQ(runtime_param_model.resources[0].value, 0u);
  EXPECT_EQ(runtime_param_model.external_buffer_bindings[1].arg_index, 1u);
  EXPECT_EQ(runtime_param_model.external_buffer_bindings[1].resource_index, 1u);
  EXPECT_EQ(runtime_param_model.resources[1].role,
            GfxMpsrtExternalBufferRole::TensorInput);
  EXPECT_TRUE(runtime_param_model.resources[1].has_tensor_value);
  EXPECT_EQ(runtime_param_model.resources[1].value, 1u);
  EXPECT_EQ(runtime_param_model.external_buffer_bindings[2].arg_index, 2u);
  EXPECT_EQ(runtime_param_model.external_buffer_bindings[2].resource_index, 2u);
  EXPECT_EQ(runtime_param_model.resources[2].role,
            GfxMpsrtExternalBufferRole::RuntimeParams);
  EXPECT_FALSE(runtime_param_model.resources[2].has_tensor_value);
  EXPECT_EQ(runtime_param_model.external_buffer_bindings[3].arg_index, 3u);
  EXPECT_EQ(runtime_param_model.external_buffer_bindings[3].resource_index, 3u);
  EXPECT_EQ(runtime_param_model.resources[3].role,
            GfxMpsrtExternalBufferRole::TensorOutput);
  EXPECT_TRUE(runtime_param_model.resources[3].has_tensor_value);
  EXPECT_EQ(runtime_param_model.resources[3].value, 3u);
  const auto *transient_resource =
      runtime_mpsrt::find_mpsrt_resource_for_value(runtime_param_model, 2u);
  ASSERT_NE(transient_resource, nullptr);
  EXPECT_EQ(transient_resource->role, GfxMpsrtExternalBufferRole::Unknown);
  EXPECT_EQ(transient_resource->lifetime,
            runtime_mpsrt::MpsrtRuntimeResourceLifetime::Transient);
  EXPECT_TRUE((transient_resource->tensor_desc.flags &
               GfxMpsrtTensorFlagExternalIo) == 0);
  EXPECT_TRUE(
      (transient_resource->tensor_desc.flags & GfxMpsrtTensorFlagConst) == 0);
}

TEST(GfxStagePolicyTest, MpsrtRuntimeModelBuildsMslDispatchStage) {
  const auto add = make_large_add_node();
  const auto plan = select_stage_optimization_plan(nullptr, GpuBackend::Metal,
                                                   "Add", add, ov::element::f16,
                                                   /*has_bias=*/false,
                                                   /*has_activation=*/false,
                                                   /*has_batchnorm=*/false, {});
  const auto stage = gfx_mpsrt_make_stage_desc(plan, "Add");
  const auto lhs = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagExternalIo);
  const auto rhs = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagExternalIo);
  const auto out = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagTransient);
  const auto builder_plan = gfx_mpsrt_make_builder_plan(
      stage, {lhs, rhs}, {out}, gfx_mpsrt_stage_record_key(stage));

  runtime_mpsrt::MpsrtModel model;
  std::string error;
  ASSERT_TRUE(runtime_mpsrt::build_mpsrt_model_from_builder_plan(
      builder_plan, model, &error))
      << error;

  ASSERT_EQ(model.tensors.size(), 3u);
  ASSERT_EQ(model.stages.size(), 1u);
  EXPECT_EQ(model.semantic_input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_EQ(model.semantic_output_values, std::vector<GfxMpsrtValue>({2u}));
  EXPECT_EQ(model.input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_EQ(model.output_values, std::vector<GfxMpsrtValue>({2u}));
  EXPECT_EQ(model.external_values, std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
  EXPECT_EQ(model.external_input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_EQ(model.external_output_values, std::vector<GfxMpsrtValue>({2u}));
  const auto &runtime_stage = model.stages.front();
  EXPECT_EQ(runtime_stage.kind, GfxMpsrtStageKind::MSLDispatch);
  EXPECT_EQ(runtime_stage.kernel_name, "eltwise_fused_buffer");
  EXPECT_EQ(runtime_stage.dispatch_kernel_family, "eltwise_fused_buffer");
  EXPECT_EQ(runtime_stage.dispatch_entry_point, "eltwise_fused_buffer");
  EXPECT_EQ(runtime_stage.dispatch_kernel_family_id,
            static_cast<uint32_t>(GfxKernelFamily::EltwiseFusedBuffer));
  EXPECT_EQ(runtime_stage.dispatch_flags,
            GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
  EXPECT_EQ(runtime_stage.dispatch_threads_per_threadgroup, 256u);
  EXPECT_TRUE(runtime_stage.dispatch_precompiled_kernel_required);
  EXPECT_EQ(runtime_stage.msl_dispatch_desc.kernel_family,
            static_cast<uint32_t>(GfxKernelFamily::EltwiseFusedBuffer));
  EXPECT_EQ(runtime_stage.msl_dispatch_desc.input_count, 2u);
  EXPECT_EQ(runtime_stage.msl_dispatch_desc.output_count, 1u);
  EXPECT_EQ(runtime_stage.msl_dispatch_desc.flags,
            GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
  EXPECT_EQ(runtime_stage.inputs, std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_EQ(runtime_stage.outputs, std::vector<GfxMpsrtValue>({2u}));
  EXPECT_EQ(runtime_stage.kernel_buffer_order,
            std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
  ASSERT_EQ(runtime_stage.output_descs.size(), 1u);
  EXPECT_EQ(runtime_stage.output_descs.front().byte_length,
            1u * 64u * 80u * 80u * 2u);
}

TEST(GfxStagePolicyTest, MpsrtRuntimeStageFromDescUsesCustomKernelManifestAbi) {
  const auto add = make_large_add_node();
  const auto plan = select_stage_optimization_plan(nullptr, GpuBackend::Metal,
                                                   "Add", add, ov::element::f16,
                                                   /*has_bias=*/false,
                                                   /*has_activation=*/false,
                                                   /*has_batchnorm=*/false, {});
  auto stage_desc = gfx_mpsrt_make_stage_desc(plan, "Add");
  stage_desc.stage_manifest.custom_kernel.external_buffer_abi =
      make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorOutput,
                                 GfxKernelBufferRole::TensorInput,
                                 GfxKernelBufferRole::TensorInput});

  const auto out = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagTransient);
  const std::vector<GfxMpsrtValue> inputs = {0u, 1u};
  const std::vector<GfxMpsrtValue> outputs = {2u};
  const auto runtime_stage =
      runtime_mpsrt::make_mpsrt_runtime_stage_from_desc(
          stage_desc, gfx_mpsrt_stage_record_key(stage_desc), inputs, outputs,
          {gfx_mpsrt_to_abi_desc(out)});

  EXPECT_EQ(runtime_stage.kind, GfxMpsrtStageKind::MSLDispatch);
  EXPECT_EQ(runtime_stage.kernel_name, "eltwise_fused_buffer");
  EXPECT_EQ(runtime_stage.dispatch_kernel_family, "eltwise_fused_buffer");
  EXPECT_EQ(runtime_stage.dispatch_entry_point, "eltwise_fused_buffer");
  EXPECT_EQ(runtime_stage.dispatch_kernel_family_id,
            static_cast<uint32_t>(GfxKernelFamily::EltwiseFusedBuffer));
  EXPECT_EQ(runtime_stage.dispatch_flags,
            GfxMpsrtMslDispatchFlagPrecompiledMetallibRequired);
  EXPECT_EQ(runtime_stage.dispatch_threads_per_threadgroup, 256u);
  EXPECT_TRUE(runtime_stage.dispatch_precompiled_kernel_required);
  EXPECT_EQ(runtime_stage.inputs, inputs);
  EXPECT_EQ(runtime_stage.outputs, outputs);
  EXPECT_EQ(runtime_stage.kernel_buffer_order,
            std::vector<GfxMpsrtValue>({2u, 0u, 1u}));
  EXPECT_EQ(runtime_stage.msl_dispatch_desc.input_count, 2u);
  EXPECT_EQ(runtime_stage.msl_dispatch_desc.output_count, 1u);
}

TEST(GfxStagePolicyTest, MpsrtRuntimeModelRejectsMalformedMslDispatch) {
  const auto add = make_large_add_node();
  const auto plan = select_stage_optimization_plan(nullptr, GpuBackend::Metal,
                                                   "Add", add, ov::element::f16,
                                                   /*has_bias=*/false,
                                                   /*has_activation=*/false,
                                                   /*has_batchnorm=*/false, {});
  const auto stage = gfx_mpsrt_make_stage_desc(plan, "Add");
  const auto lhs = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagExternalIo);
  const auto rhs = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagExternalIo);
  const auto out = gfx_mpsrt_make_tensor_desc({1, 64, 80, 80}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagTransient);
  auto builder_plan = gfx_mpsrt_make_builder_plan(
      stage, {lhs, rhs}, {out}, gfx_mpsrt_stage_record_key(stage));
  ASSERT_EQ(builder_plan.records.size(), 5u);
  builder_plan.records[3].msl_dispatch_desc.kernel_family = 0;

  runtime_mpsrt::MpsrtModel model;
  std::string error;
  EXPECT_FALSE(
      runtime_mpsrt::build_mpsrt_model_from_builder_plan(
          builder_plan, model, &error));
  EXPECT_NE(error.find("MSL dispatch kernel family is not set"),
            std::string::npos);
}

TEST(GfxStagePolicyTest,
     MpsrtRuntimeModelRejectsMalformedStorageBridgeContract) {
  const auto pool = make_aligned_maxpool_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "MaxPool", pool, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  auto stage = gfx_mpsrt_make_stage_desc(plan, "MaxPool");
  const auto input = gfx_mpsrt_make_tensor_desc(
      {1, 4, 16, 16}, ov::element::f16, GfxStageStorageKind::Image,
      GfxMpsrtTensorFlagExternalIo);
  const auto output = gfx_mpsrt_make_tensor_desc({1, 4, 8, 8}, ov::element::f16,
                                                 GfxStageStorageKind::Image,
                                                 GfxMpsrtTensorFlagTransient);
  auto builder_plan = gfx_mpsrt_make_builder_plan(
      stage, {input}, {output}, gfx_mpsrt_stage_record_key(stage));
  ASSERT_EQ(builder_plan.storage_bridges.size(), 2u);
  builder_plan.storage_bridges.front().direction =
      GfxMpsrtStorageBridgeDirection::Unknown;

  runtime_mpsrt::MpsrtModel model;
  std::string error;
  EXPECT_FALSE(
      runtime_mpsrt::build_mpsrt_model_from_builder_plan(
          builder_plan, model, &error));
  EXPECT_NE(error.find("storage bridge contract is invalid"),
            std::string::npos);
}

TEST(GfxStagePolicyTest, MpsrtRuntimeModelAdaptsExpandedExternalBufferAbi) {
  const auto add = make_large_add_node();
  const auto plan = select_stage_optimization_plan(nullptr, GpuBackend::Metal,
                                                   "Add", add, ov::element::f16,
                                                   /*has_bias=*/false,
                                                   /*has_activation=*/false,
                                                   /*has_batchnorm=*/false, {});
  const auto stage = gfx_mpsrt_make_stage_desc(plan, "Add");
  const auto lhs = gfx_mpsrt_make_tensor_desc({1, 64}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagExternalIo);
  const auto out = gfx_mpsrt_make_tensor_desc({1, 64}, ov::element::f16,
                                              GfxStageStorageKind::Buffer,
                                              GfxMpsrtTensorFlagTransient);
  const auto builder_plan = gfx_mpsrt_make_builder_plan(
      stage, {lhs}, {out}, gfx_mpsrt_stage_record_key(stage));

  runtime_mpsrt::MpsrtModel model;
  std::string error;
  ASSERT_TRUE(runtime_mpsrt::build_mpsrt_model_from_builder_plan(
      builder_plan, model, &error))
      << error;
  ASSERT_TRUE(
      runtime_mpsrt::adapt_mpsrt_model_to_external_buffer_abi(
          model, 3u, 1u, &error))
      << error;

  EXPECT_EQ(model.semantic_input_values, std::vector<GfxMpsrtValue>({0u}));
  EXPECT_EQ(model.semantic_output_values, std::vector<GfxMpsrtValue>({1u}));
  EXPECT_EQ(model.input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_EQ(model.output_values, std::vector<GfxMpsrtValue>({2u}));
  EXPECT_EQ(model.external_values, std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
  EXPECT_EQ(model.external_input_values, std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_EQ(model.external_output_values, std::vector<GfxMpsrtValue>({2u}));
  EXPECT_EQ(model.external_buffer_roles,
            std::vector<GfxMpsrtExternalBufferRole>(
                {GfxMpsrtExternalBufferRole::TensorInput,
                 GfxMpsrtExternalBufferRole::TensorInput,
                 GfxMpsrtExternalBufferRole::TensorOutput}));
  ASSERT_EQ(model.stages.size(), 1u);
  EXPECT_EQ(model.stages.front().inputs, std::vector<GfxMpsrtValue>({0u, 1u}));
  EXPECT_EQ(model.stages.front().outputs, std::vector<GfxMpsrtValue>({2u}));
  EXPECT_EQ(model.stages.front().kernel_buffer_order,
            std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
  EXPECT_EQ(model.stages.front().msl_dispatch_desc.input_count, 2u);
  EXPECT_EQ(model.stages.front().msl_dispatch_desc.output_count, 1u);
}

TEST(GfxStagePolicyTest,
     MpsrtRuntimeModelAdaptsExternalBufferRolesFromExplicitManifestOrder) {
  const auto add = make_large_add_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Metal, "Softmax", add, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  const auto stage = gfx_mpsrt_make_stage_desc(plan, "Softmax");
  const auto input = gfx_mpsrt_make_tensor_desc({1, 64}, ov::element::f16,
                                                GfxStageStorageKind::Buffer,
                                                GfxMpsrtTensorFlagExternalIo);
  const auto output = gfx_mpsrt_make_tensor_desc({1, 64}, ov::element::f16,
                                                 GfxStageStorageKind::Buffer,
                                                 GfxMpsrtTensorFlagTransient);
  auto builder_plan = gfx_mpsrt_make_builder_plan(
      stage, {input}, {output}, gfx_mpsrt_stage_record_key(stage));
  builder_plan.external_buffer_roles = {
      GfxMpsrtExternalBufferRole::TensorInput,
      GfxMpsrtExternalBufferRole::TensorOutput,
      GfxMpsrtExternalBufferRole::RuntimeParams};

  runtime_mpsrt::MpsrtModel model;
  std::string error;
  ASSERT_TRUE(runtime_mpsrt::build_mpsrt_model_from_builder_plan(
      builder_plan, model, &error))
      << error;
  ASSERT_TRUE(
      runtime_mpsrt::adapt_mpsrt_model_to_external_buffer_abi(
          model, 3u, 1u, &error))
      << error;

  EXPECT_EQ(model.input_values, std::vector<GfxMpsrtValue>({0u, 2u}));
  EXPECT_EQ(model.output_values, std::vector<GfxMpsrtValue>({1u}));
  EXPECT_EQ(model.external_values, std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
  EXPECT_EQ(model.external_input_values, std::vector<GfxMpsrtValue>({0u, 2u}));
  EXPECT_EQ(model.external_output_values, std::vector<GfxMpsrtValue>({1u}));
  EXPECT_EQ(model.external_buffer_roles,
            std::vector<GfxMpsrtExternalBufferRole>(
                {GfxMpsrtExternalBufferRole::TensorInput,
                 GfxMpsrtExternalBufferRole::TensorOutput,
                 GfxMpsrtExternalBufferRole::RuntimeParams}));
  ASSERT_EQ(model.stages.size(), 1u);
  EXPECT_EQ(model.stages.front().inputs, std::vector<GfxMpsrtValue>({0u, 2u}));
  EXPECT_EQ(model.stages.front().outputs, std::vector<GfxMpsrtValue>({1u}));
  EXPECT_EQ(model.stages.front().kernel_buffer_order,
            std::vector<GfxMpsrtValue>({0u, 1u, 2u}));
  EXPECT_EQ(model.stages.front().msl_dispatch_desc.input_count, 2u);
  EXPECT_EQ(model.stages.front().msl_dispatch_desc.output_count, 1u);
}

TEST(GfxStagePolicyTest,
     MpsrtRuntimeModelEnumeratesExternalTensorBindingsForRuntimeBoundary) {
  runtime_mpsrt::MpsrtModel model;
  auto image_desc = gfx_mpsrt_make_tensor_desc(
      {1, 4, 8, 8}, ov::element::f16, GfxStageStorageKind::Image,
      GfxMpsrtTensorFlagExternalIo);
  const auto image_abi_desc = gfx_mpsrt_to_abi_desc(image_desc);
  runtime_mpsrt::MpsrtRuntimeTensor input_tensor;
  input_tensor.value = 0u;
  input_tensor.desc = image_abi_desc;
  runtime_mpsrt::MpsrtRuntimeTensor output_tensor;
  output_tensor.value = 1u;
  output_tensor.desc = image_abi_desc;
  model.tensors = {input_tensor, output_tensor};
  model.external_values = {0u, 1u};
  model.external_output_values = {1u};
  model.external_buffer_roles = {GfxMpsrtExternalBufferRole::TensorInput,
                                 GfxMpsrtExternalBufferRole::TensorOutput};

  std::vector<runtime_mpsrt::MpsrtExternalTensorBinding> bindings;
  std::string error;
  ASSERT_TRUE(
      runtime_mpsrt::mpsrt_model_external_tensor_bindings(model, bindings,
                                                          &error))
      << error;
  ASSERT_EQ(bindings.size(), 2u);
  EXPECT_EQ(bindings[0].arg_index, 0u);
  EXPECT_EQ(bindings[0].value, 0u);
  EXPECT_EQ(bindings[0].role, GfxMpsrtExternalBufferRole::TensorInput);
  EXPECT_EQ(bindings[0].bridge_direction,
            GfxMpsrtStorageBridgeDirection::BufferToImage);
  EXPECT_EQ(bindings[1].arg_index, 1u);
  EXPECT_EQ(bindings[1].value, 1u);
  EXPECT_EQ(bindings[1].role, GfxMpsrtExternalBufferRole::TensorOutput);
  EXPECT_EQ(bindings[1].bridge_direction,
            GfxMpsrtStorageBridgeDirection::ImageToBuffer);

  runtime_mpsrt::MpsrtModel explicit_model = model;
  explicit_model.external_values = {0u, 1u};
  explicit_model.external_buffer_bindings = {{0u, 0u}, {1u, 1u}, {2u, 2u}};
  runtime_mpsrt::MpsrtRuntimeResource input_resource;
  input_resource.resource_index = 0u;
  input_resource.role = GfxMpsrtExternalBufferRole::TensorInput;
  input_resource.lifetime =
      runtime_mpsrt::MpsrtRuntimeResourceLifetime::External;
  input_resource.arg_index = 0u;
  input_resource.has_tensor_value = true;
  input_resource.value = 0u;
  input_resource.tensor_desc = image_abi_desc;
  runtime_mpsrt::MpsrtRuntimeResource runtime_params_resource;
  runtime_params_resource.resource_index = 1u;
  runtime_params_resource.role = GfxMpsrtExternalBufferRole::RuntimeParams;
  runtime_params_resource.lifetime =
      runtime_mpsrt::MpsrtRuntimeResourceLifetime::External;
  runtime_params_resource.arg_index = 1u;
  runtime_mpsrt::MpsrtRuntimeResource output_resource;
  output_resource.resource_index = 2u;
  output_resource.role = GfxMpsrtExternalBufferRole::TensorOutput;
  output_resource.lifetime =
      runtime_mpsrt::MpsrtRuntimeResourceLifetime::External;
  output_resource.arg_index = 2u;
  output_resource.has_tensor_value = true;
  output_resource.value = 1u;
  output_resource.tensor_desc = image_abi_desc;
  explicit_model.resources = {input_resource,
                              runtime_params_resource,
                              output_resource};

  ASSERT_TRUE(runtime_mpsrt::mpsrt_model_external_tensor_bindings(
      explicit_model, bindings, &error))
      << error;
  ASSERT_EQ(bindings.size(), 2u);
  EXPECT_EQ(bindings[0].arg_index, 0u);
  EXPECT_EQ(bindings[0].value, 0u);
  EXPECT_EQ(bindings[1].arg_index, 2u);
  EXPECT_EQ(bindings[1].value, 1u);
  EXPECT_EQ(bindings[1].bridge_direction,
            GfxMpsrtStorageBridgeDirection::ImageToBuffer);

  std::vector<runtime_mpsrt::MpsrtTensorBindingPlanEntry> binding_plan;
  ASSERT_TRUE(runtime_mpsrt::mpsrt_model_tensor_binding_plan(
      explicit_model, binding_plan, &error))
      << error;
  ASSERT_EQ(binding_plan.size(), 3u);
  EXPECT_EQ(binding_plan[0].arg_index, 0u);
  EXPECT_TRUE(binding_plan[0].has_tensor_value);
  EXPECT_EQ(binding_plan[0].value, 0u);
  EXPECT_EQ(binding_plan[1].arg_index, 1u);
  EXPECT_EQ(binding_plan[1].role, GfxMpsrtExternalBufferRole::RuntimeParams);
  EXPECT_FALSE(binding_plan[1].has_tensor_value);
  EXPECT_EQ(binding_plan[2].arg_index, 2u);
  EXPECT_TRUE(binding_plan[2].has_tensor_value);
  EXPECT_EQ(binding_plan[2].value, 1u);
}

TEST(GfxStagePolicyTest, VulkanConvolutionAllowsOnlyReluActivationFusion) {
  EXPECT_TRUE(allow_stage_activation_fusion(GpuBackend::Vulkan, "Convolution",
                                            ActivationKind::Relu));
  EXPECT_FALSE(allow_stage_activation_fusion(GpuBackend::Vulkan, "Convolution",
                                             ActivationKind::Sigmoid));
  EXPECT_FALSE(allow_stage_activation_fusion(
      GpuBackend::Vulkan, "GroupConvolution", ActivationKind::Relu));
}

TEST(GfxStagePolicyTest,
     VulkanGroupConvolutionStaysConservativeWithBiasOrActivation) {
  const auto gconv = make_depthwise_group_conv_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Vulkan, "GroupConvolution", gconv, ov::element::f16,
      /*has_bias=*/true,
      /*has_activation=*/true,
      /*has_batchnorm=*/false, {});

  EXPECT_FALSE(plan.post_ops.bias);
  EXPECT_FALSE(plan.post_ops.activation);
  EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::None);
  EXPECT_EQ(plan.conv.algorithm.kind, GfxConvAlgorithmKind::None);
}

TEST(GfxStagePolicyTest, VulkanMatMulUsesAdaptiveSubmitWindow) {
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Vulkan, "MatMul", nullptr, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});

  EXPECT_EQ(plan.archetype, GfxStageArchetype::MatMul);
  EXPECT_FALSE(plan.execution.submit.isolate);
  EXPECT_GE(plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest,
     VulkanLargeBinaryChunkedStageUsesIsolatedSubmitWindow) {
  const auto add = make_large_add_node();
  GfxStageRuntimeTraits traits{};
  traits.binary_chunked = true;
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Vulkan, "Add", add, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, traits);

  EXPECT_EQ(plan.archetype, GfxStageArchetype::BinaryElementwise);
  EXPECT_TRUE(plan.execution.submit.isolate);
  EXPECT_GE(plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest, VulkanLargeConcatUsesIsolatedSubmitWindow) {
  const auto concat = make_large_concat_node();
  GfxStageRuntimeTraits traits{};
  traits.split_concat_chunked = true;
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Vulkan, "Concat", concat, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, traits);

  EXPECT_EQ(plan.archetype, GfxStageArchetype::SplitConcat);
  EXPECT_TRUE(plan.execution.submit.isolate);
  EXPECT_GE(plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest,
     VulkanConstrainedLargeBinaryChunkedStageCanShareSubmitWindow) {
  GpuExecutionDeviceInfo info;
  info.backend = GpuBackend::Vulkan;
  info.device_key = "test-rpi";
  info.preferred_simd_width = 16u;
  info.subgroup_size = 16u;
  info.max_total_threads_per_group = 256u;
  info.max_threads_per_group = {256u, 256u, 64u};
  FakeDeviceInfoBufferManager buffer_manager(info);
  const auto add = make_large_add_node();
  GfxStageRuntimeTraits traits{};
  traits.binary_chunked = true;
  const auto plan = select_stage_optimization_plan(
      &buffer_manager, GpuBackend::Vulkan, "Add", add, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, traits);

  EXPECT_EQ(plan.archetype, GfxStageArchetype::BinaryElementwise);
  EXPECT_FALSE(plan.execution.submit.isolate);
  EXPECT_GE(plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest, VulkanConstrainedLargeConcatCanShareSubmitWindow) {
  GpuExecutionDeviceInfo info;
  info.backend = GpuBackend::Vulkan;
  info.device_key = "test-rpi";
  info.preferred_simd_width = 16u;
  info.subgroup_size = 16u;
  info.max_total_threads_per_group = 256u;
  info.max_threads_per_group = {256u, 256u, 64u};
  FakeDeviceInfoBufferManager buffer_manager(info);
  const auto concat = make_large_concat_node();
  GfxStageRuntimeTraits traits{};
  traits.split_concat_chunked = true;
  const auto plan = select_stage_optimization_plan(
      &buffer_manager, GpuBackend::Vulkan, "Concat", concat, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, traits);

  EXPECT_EQ(plan.archetype, GfxStageArchetype::SplitConcat);
  EXPECT_FALSE(plan.execution.submit.isolate);
  EXPECT_EQ(plan.execution.submit.weight, 4u);
}

TEST(GfxStagePolicyTest, VulkanLargeTransposeUsesChunkedSubmitTraits) {
  const auto transpose = make_large_transpose_node();
  GfxStageRuntimeTraits traits{};
  traits.transpose_chunked = true;
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Vulkan, "Transpose", transpose, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, traits);

  EXPECT_EQ(plan.archetype, GfxStageArchetype::Layout);
  EXPECT_FALSE(plan.execution.submit.isolate);
  EXPECT_EQ(plan.execution.submit.weight, 4u);
}

TEST(GfxStagePolicyTest, VulkanChunkedConvolutionUsesIsolatedSubmitWindow) {
  const auto conv = make_large_chunked_conv_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Vulkan, "Convolution", conv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});

  EXPECT_EQ(plan.archetype, GfxStageArchetype::Convolution);
  EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::Chunked);
  EXPECT_TRUE(plan.execution.submit.isolate);
  EXPECT_GE(plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest,
     VulkanMediumChunkedConvolutionUsesIsolatedSubmitWindow) {
  const auto conv = make_medium_chunked_conv_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Vulkan, "Convolution", conv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});

  EXPECT_EQ(plan.archetype, GfxStageArchetype::Convolution);
  EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::Chunked);
  EXPECT_TRUE(plan.execution.submit.isolate);
  EXPECT_GE(plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest,
     VulkanLightDepthwiseChunkedConvolutionUsesIsolatedSubmitWindow) {
  const auto gconv = make_light_depthwise_group_conv_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Vulkan, "GroupConvolution", gconv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});

  EXPECT_EQ(plan.archetype, GfxStageArchetype::GroupConvolution);
  EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::GroupChunked);
  EXPECT_TRUE(plan.execution.submit.isolate);
  EXPECT_GE(plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest,
     VulkanLightSpatial3x3ConvolutionFallsBackToSharedMlirRoute) {
  const auto conv = make_light_spatial3x3_conv_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Vulkan, "Convolution", conv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});

  EXPECT_EQ(plan.archetype, GfxStageArchetype::Convolution);
  EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::None);
  EXPECT_EQ(plan.conv.algorithm.kind, GfxConvAlgorithmKind::None);
  EXPECT_TRUE(plan.execution.submit.isolate);
}

TEST(GfxStagePolicyTest,
     VulkanSerialConvolutionChainCanShareAdaptiveSubmitWindow) {
  const auto [conv0, conv1] = make_light_spatial3x3_conv_chain();

  const auto first_plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Vulkan, "Convolution", conv0, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});
  const auto second_plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Vulkan, "Convolution", conv1, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});

  EXPECT_EQ(first_plan.conv.kind, GfxConvRouteKind::None);
  EXPECT_EQ(second_plan.conv.kind, GfxConvRouteKind::None);
  EXPECT_FALSE(first_plan.execution.submit.isolate);
  EXPECT_FALSE(second_plan.execution.submit.isolate);
  EXPECT_GE(first_plan.execution.submit.weight, 8u);
  EXPECT_GE(second_plan.execution.submit.weight, 8u);
}

TEST(GfxStagePolicyTest, VulkanConcatAdjacentConvolutionRemainsIsolated) {
  const auto conv = make_concat_fed_spatial3x3_conv_node();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Vulkan, "Convolution", conv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});

  EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::None);
  EXPECT_TRUE(plan.execution.submit.isolate);
}

TEST(GfxStagePolicyTest,
     VulkanLayoutAdjacentChunkedConvolutionRemainsIsolated) {
  const auto conv = make_chunked_conv_with_reshape_consumer();
  const auto plan = select_stage_optimization_plan(
      nullptr, GpuBackend::Vulkan, "Convolution", conv, ov::element::f16,
      /*has_bias=*/false,
      /*has_activation=*/false,
      /*has_batchnorm=*/false, {});

  EXPECT_EQ(plan.conv.kind, GfxConvRouteKind::Chunked);
  EXPECT_TRUE(plan.execution.submit.isolate);
}
