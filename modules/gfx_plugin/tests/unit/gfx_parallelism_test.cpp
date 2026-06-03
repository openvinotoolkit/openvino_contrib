// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include <utility>

#include "gtest/gtest.h"

#include "kernel_ir/gfx_kernel_dispatch.hpp"
#include "runtime/gpu_buffer_manager.hpp"
#include "runtime/gpu_device_info.hpp"
#include "runtime/gfx_parallelism.hpp"

namespace {

class TestDeviceInfoBufferManager final
    : public ov::gfx_plugin::GpuBufferManager {
public:
  explicit TestDeviceInfoBufferManager(
      ov::gfx_plugin::GpuExecutionDeviceInfo info)
      : m_info(std::move(info)) {}

  std::optional<ov::gfx_plugin::GpuExecutionDeviceInfo>
  query_execution_device_info() const override {
    return m_info;
  }

private:
  ov::gfx_plugin::GpuExecutionDeviceInfo m_info;
};

ov::gfx_plugin::GfxParallelismCaps make_caps() {
  ov::gfx_plugin::GfxParallelismCaps caps;
  caps.profile_key = "test:opencl-profile";
  caps.preferred_simd_width = 32;
  caps.subgroup_size = 32;
  caps.max_total_threads_per_group = 64;
  caps.max_threads_per_group = {64, 64, 1};
  caps.chunk_dispatch = ov::gfx_plugin::make_opencl_chunk_dispatch_profile();
  return caps;
}

ov::gfx_plugin::GfxParallelismCaps make_broadcom_caps() {
  auto caps = make_caps();
  caps.profile_key = "test:broadcom-v3d-profile";
  caps.preferred_simd_width = 16;
  caps.subgroup_size = 16;
  caps.enable_skinny_matmul_tiles = true;
  caps.chunk_dispatch.retune_threads_to_workload = true;
  caps.scale_conv_threads_for_large_spatial = true;
  caps.scale_conv_threads_for_dense_reduction = true;
  caps.scale_conv_threads_for_pointwise_reduction = true;
  caps.conv_spatial_micro_tile_requires_large_output_area = true;
  return caps;
}

ov::gfx_plugin::GfxParallelismCaps make_large_broadcom_caps() {
  auto caps = make_broadcom_caps();
  caps.max_total_threads_per_group = 256;
  caps.max_threads_per_group = {256, 256, 1};
  return caps;
}

ov::gfx_plugin::GfxParallelismCaps make_broadcom_channel_blocking_caps() {
  auto caps = make_large_broadcom_caps();
  caps.profile_key = "test:broadcom-v3d-profile:channel-blocking";
  caps.supports_conv_output_channel_blocking = true;
  return caps;
}

ov::gfx_plugin::GfxParallelismCaps make_channel_blocking_caps() {
  ov::gfx_plugin::GfxParallelismCaps caps;
  caps.profile_key = "test:metal-profile:channel-blocking";
  caps.preferred_simd_width = 32;
  caps.subgroup_size = 32;
  caps.max_total_threads_per_group = 256;
  caps.max_threads_per_group = {256, 256, 1};
  caps.supports_conv_output_channel_blocking = true;
  caps.supports_conv_channel_block_spatial_tiling = true;
  caps.sort_matmul_tiles_by_shape = false;
  caps.chunk_dispatch = ov::gfx_plugin::make_metal_chunk_dispatch_profile();
  return caps;
}

} // namespace

TEST(GfxParallelism,
     SelectMatMulParallelismPrefersSquareTilesForSquareOutputs) {
  const auto plan = ov::gfx_plugin::select_matmul_parallelism(
      make_caps(), ov::Shape{1, 256, 256});

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.dispatch.threads_h, 8u);
  EXPECT_EQ(plan.dispatch.threads_w, 8u);
}

TEST(GfxParallelism, QueryParallelismCapsUsesBackendNeutralDeviceInfo) {
  ov::gfx_plugin::GpuExecutionDeviceInfo info;
  info.backend = ov::gfx_plugin::GpuBackend::OpenCL;
  info.device_key = "test-device";
  info.preferred_simd_width = 16;
  info.subgroup_size = 32;
  info.max_total_threads_per_group = 128;
  info.max_threads_per_group = {128, 32, 8};
  info.supports_conv_output_channel_blocking = true;
  info.supports_conv_channel_block_spatial_tiling = true;
  info.parallelism_profile.profile_key = "test-device-parallelism";
  info.parallelism_profile.enable_skinny_matmul_tiles = true;

  TestDeviceInfoBufferManager buffer_manager(info);
  const auto caps = ov::gfx_plugin::query_parallelism_caps(&buffer_manager);

  EXPECT_EQ(caps.profile_key, "test-device-parallelism");
  EXPECT_EQ(caps.preferred_simd_width, 16u);
  EXPECT_EQ(caps.subgroup_size, 32u);
  EXPECT_EQ(caps.max_total_threads_per_group, 128u);
  EXPECT_EQ(caps.max_threads_per_group[0], 128u);
  EXPECT_EQ(caps.max_threads_per_group[1], 32u);
  EXPECT_EQ(caps.max_threads_per_group[2], 8u);
  EXPECT_TRUE(caps.supports_conv_output_channel_blocking);
  EXPECT_TRUE(caps.supports_conv_channel_block_spatial_tiling);
  EXPECT_TRUE(caps.enable_skinny_matmul_tiles);
}

TEST(GfxParallelism, SelectMatMulParallelismPrefersWideTilesForWideOutputs) {
  const auto plan = ov::gfx_plugin::select_matmul_parallelism(
      make_caps(), ov::Shape{128, 1600});

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_TRUE(plan.dispatch.enabled);
  EXPECT_GT(plan.dispatch.threads_h, 0u);
  EXPECT_GT(plan.dispatch.threads_w, 0u);
  EXPECT_GE(plan.dispatch.threads_w, plan.dispatch.threads_h);
}

TEST(GfxParallelism,
     SelectMatMulParallelismUsesSkinnyTilesForWideBroadcomOutputs) {
  const auto plan = ov::gfx_plugin::select_matmul_parallelism(
      make_broadcom_caps(), ov::Shape{128, 1600});

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_TRUE(plan.dispatch.enabled);
  EXPECT_LE(plan.dispatch.threads_h, 2u);
  EXPECT_GE(plan.dispatch.threads_w, 8u);
}

TEST(GfxParallelism, SelectMatMulParallelismKeepsBroadcomSquareOutputsDense) {
  const auto plan = ov::gfx_plugin::select_matmul_parallelism(
      make_broadcom_caps(), ov::Shape{1, 1024, 1024});

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_TRUE(plan.dispatch.enabled);
  EXPECT_EQ(plan.dispatch.threads_h, plan.dispatch.threads_w);
  EXPECT_GE(plan.dispatch.threads_h * plan.dispatch.threads_w, 32u);
}

TEST(GfxParallelism,
     SelectBroadcomConvParallelismUsesDenserStride1Threadgroups) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_broadcom_caps(), ov::Shape{1, 64, 80, 80}, 64, 64, 64 * 3 * 3, false,
      false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_TRUE(plan.dispatch.enabled);
  EXPECT_GE(plan.dispatch.threads_h * plan.dispatch.threads_w, 64u);
}

TEST(GfxParallelism,
     SelectBroadcomConvParallelismUsesDenserStride2Threadgroups) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_broadcom_caps(), ov::Shape{1, 64, 80, 80}, 64, 64, 64 * 3 * 3, true,
      false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_TRUE(plan.dispatch.enabled);
  EXPECT_GE(plan.dispatch.threads_h * plan.dispatch.threads_w, 64u);
}

TEST(GfxParallelism,
     SelectBroadcomConvParallelismUsesWiderThreadgroupsForHugeSpatialOutputs) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_large_broadcom_caps(), ov::Shape{1, 16, 320, 320}, 16, 32,
      16 * 3 * 3, false, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_TRUE(plan.dispatch.enabled);
  EXPECT_GE(plan.dispatch.threads_h * plan.dispatch.threads_w, 64u);
}

TEST(GfxParallelism,
     SelectBroadcomPointwiseConvKeepsOccupancyHeadroomForHugeSpatialOutputs) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_large_broadcom_caps(), ov::Shape{1, 48, 160, 160}, 48, 48, 48, false,
      false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_TRUE(plan.dispatch.enabled);
  EXPECT_EQ(plan.dispatch.threads_h * plan.dispatch.threads_w, 64u);
}

TEST(GfxParallelism,
     SelectBroadcomDensePointwiseConvUsesOccupancyDenseThreadgroups) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_large_broadcom_caps(), ov::Shape{1, 384, 160, 160}, 384, 384, 384,
      false, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_TRUE(plan.dispatch.enabled);
  EXPECT_EQ(plan.dispatch.threads_h * plan.dispatch.threads_w, 128u);
}

TEST(GfxParallelism,
     SelectBroadcomConvParallelismAvoidsFullThreadgroupForUltraDenseWorkloads) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_large_broadcom_caps(), ov::Shape{1, 256, 80, 80}, 256, 256,
      256 * 3 * 3, false, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_TRUE(plan.dispatch.enabled);
  EXPECT_EQ(plan.dispatch.threads_h * plan.dispatch.threads_w, 128u);
}

TEST(
    GfxParallelism,
    SelectBroadcomDenseConvKeepsScalarChannelsUntilBackendCapabilityIsEnabled) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_large_broadcom_caps(), ov::Shape{1, 256, 80, 80}, 256, 256,
      256 * 3 * 3, false, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.output_channel_block, 1u);
  EXPECT_EQ(plan.dispatch.channel_block, 1u);
}

TEST(GfxParallelism,
     SelectBroadcomCapabilityEnabledDenseConvUsesFusedOutputChannelBlockWhenAccumulatorBudgetFits) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_broadcom_channel_blocking_caps(), ov::Shape{1, 256, 80, 80}, 256,
      256, 256 * 3 * 3, false, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.output_channel_block, 8u);
  EXPECT_EQ(plan.dispatch.channel_block, 8u);
  EXPECT_EQ(plan.channel_block_accumulation,
            ov::gfx_plugin::ConvChannelBlockAccumulation::Fused);
  EXPECT_EQ(plan.dispatch.tile_h, plan.dispatch.threads_h);
  EXPECT_EQ(plan.dispatch.tile_w, plan.dispatch.threads_w);
}

TEST(GfxParallelism,
     SelectBroadcomCapabilityEnabledLargeStride2KeepsOc4WithoutMicroTile) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_broadcom_channel_blocking_caps(), ov::Shape{1, 192, 160, 160}, 96,
      192, 96 * 3 * 3, true, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.output_channel_block, 4u);
  EXPECT_EQ(plan.dispatch.channel_block, 4u);
  EXPECT_EQ(plan.channel_block_accumulation,
            ov::gfx_plugin::ConvChannelBlockAccumulation::Fused);
  EXPECT_EQ(plan.dispatch.tile_h, plan.dispatch.threads_h);
  EXPECT_EQ(plan.dispatch.tile_w, plan.dispatch.threads_w);
}

TEST(GfxParallelism,
     SelectBroadcomCapabilityEnabledDensePointwiseUsesOc8WithoutMicroTile) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_broadcom_channel_blocking_caps(), ov::Shape{1, 384, 160, 160}, 384,
      384, 384, false, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.output_channel_block, 8u);
  EXPECT_EQ(plan.dispatch.channel_block, 8u);
  EXPECT_EQ(plan.channel_block_accumulation,
            ov::gfx_plugin::ConvChannelBlockAccumulation::Fused);
  EXPECT_EQ(plan.dispatch.tile_h, plan.dispatch.threads_h);
  EXPECT_EQ(plan.dispatch.tile_w, plan.dispatch.threads_w);
}

TEST(GfxParallelism,
     SelectBroadcomCapabilityEnabledDenseConvKeepsFusedAccumulationWithSpatialMicroTile) {
  auto caps = make_broadcom_channel_blocking_caps();
  caps.supports_conv_channel_block_spatial_tiling = true;
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      caps, ov::Shape{1, 256, 80, 80}, 256, 256, 256 * 3 * 3, false, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.output_channel_block, 4u);
  EXPECT_EQ(plan.dispatch.channel_block, 4u);
  EXPECT_EQ(plan.channel_block_accumulation,
            ov::gfx_plugin::ConvChannelBlockAccumulation::Fused);
  EXPECT_GT(plan.dispatch.tile_h * plan.dispatch.tile_w,
            plan.dispatch.threads_h * plan.dispatch.threads_w);
}

TEST(GfxParallelism,
     SelectBroadcomCapabilityEnabledCompactDenseConvKeepsChannelOnlyReuse) {
  auto caps = make_broadcom_channel_blocking_caps();
  caps.supports_conv_channel_block_spatial_tiling = true;
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      caps, ov::Shape{1, 192, 40, 40}, 192, 192, 192 * 3 * 3, false, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.output_channel_block, 8u);
  EXPECT_EQ(plan.dispatch.channel_block, 8u);
  EXPECT_EQ(plan.channel_block_accumulation,
            ov::gfx_plugin::ConvChannelBlockAccumulation::Fused);
  EXPECT_EQ(plan.dispatch.tile_h, plan.dispatch.threads_h);
  EXPECT_EQ(plan.dispatch.tile_w, plan.dispatch.threads_w);
}

TEST(GfxParallelism,
     SelectBroadcomCapabilityEnabledStride2DenseConvUsesAccumulatorBudgetBlock) {
  auto caps = make_broadcom_channel_blocking_caps();
  caps.supports_conv_channel_block_spatial_tiling = true;
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      caps, ov::Shape{1, 768, 40, 40}, 768, 768, 768 * 3 * 3, true, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.output_channel_block, 8u);
  EXPECT_EQ(plan.dispatch.channel_block, 8u);
  EXPECT_EQ(plan.channel_block_accumulation,
            ov::gfx_plugin::ConvChannelBlockAccumulation::Fused);
  EXPECT_EQ(plan.dispatch.tile_h, plan.dispatch.threads_h);
  EXPECT_EQ(plan.dispatch.tile_w, plan.dispatch.threads_w);
}

TEST(GfxParallelism,
     SelectBroadcomCapabilityEnabledCompactMediumDenseConvKeepsChannelOnlyReuse) {
  auto caps = make_broadcom_channel_blocking_caps();
  caps.supports_conv_channel_block_spatial_tiling = true;
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      caps, ov::Shape{1, 192, 40, 40}, 192, 192, 192 * 3 * 3, true, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.output_channel_block, 8u);
  EXPECT_EQ(plan.dispatch.channel_block, 8u);
  EXPECT_EQ(plan.dispatch.tile_h, plan.dispatch.threads_h);
  EXPECT_EQ(plan.dispatch.tile_w, plan.dispatch.threads_w);
}

TEST(GfxParallelism,
     SelectBroadcomCapabilityEnabledLightConvKeepsFusedOutputChannelBlock) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_broadcom_channel_blocking_caps(), ov::Shape{1, 64, 160, 160}, 64,
      64, 64, false, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.output_channel_block, 4u);
  EXPECT_EQ(plan.dispatch.channel_block, 4u);
  EXPECT_EQ(plan.channel_block_accumulation,
            ov::gfx_plugin::ConvChannelBlockAccumulation::Fused);
}

TEST(
    GfxParallelism,
    SelectCapabilityEnabledDenseConvUsesHardwareRelativeOutputChannelBlocking) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_channel_blocking_caps(), ov::Shape{1, 256, 80, 80}, 256, 256,
      256 * 3 * 3, false, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.output_channel_block, 8u);
  EXPECT_EQ(plan.dispatch.channel_block, 8u);
  EXPECT_EQ(plan.channel_block_accumulation,
            ov::gfx_plugin::ConvChannelBlockAccumulation::Fused);
}

TEST(
    GfxParallelism,
    SelectCapabilityEnabledDenseConvKeepsOneSpatialOutputWithoutMicroTileCapability) {
  auto caps = make_channel_blocking_caps();
  caps.profile_key = "test:profile:channel-blocking-no-microtile";
  caps.supports_conv_channel_block_spatial_tiling = false;
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      caps, ov::Shape{1, 256, 80, 80}, 256, 256, 256 * 3 * 3, false, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.output_channel_block, 8u);
  EXPECT_EQ(plan.dispatch.channel_block, 8u);
  EXPECT_EQ(plan.dispatch.tile_h, plan.dispatch.threads_h);
  EXPECT_EQ(plan.dispatch.tile_w, plan.dispatch.threads_w);
}

TEST(
    GfxParallelism,
    SelectCapabilityEnabledDenseConvUsesSpatialMicroTilesFromAccumulatorBudget) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_channel_blocking_caps(), ov::Shape{1, 256, 80, 80}, 256, 256,
      256 * 3 * 3, false, false);

  ASSERT_TRUE(plan.prefer_parallel);
  const uint32_t thread_lanes =
      plan.dispatch.threads_h * plan.dispatch.threads_w;
  const uint32_t tile_lanes = plan.dispatch.tile_h * plan.dispatch.tile_w;
  EXPECT_GT(tile_lanes, thread_lanes);
  EXPECT_EQ(tile_lanes % thread_lanes, 0u);
  EXPECT_LE(tile_lanes, 4u * thread_lanes);
}

TEST(
    GfxParallelism,
    SelectCapabilityEnabledCompactDenseConvKeepsSpatialMicroTilesOutsideBroadcomGuard) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_channel_blocking_caps(), ov::Shape{1, 192, 40, 40}, 192, 192,
      192 * 3 * 3, false, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.output_channel_block, 8u);
  EXPECT_EQ(plan.dispatch.channel_block, 8u);
  EXPECT_EQ(plan.channel_block_accumulation,
            ov::gfx_plugin::ConvChannelBlockAccumulation::Fused);
  EXPECT_GT(plan.dispatch.tile_h * plan.dispatch.tile_w,
            plan.dispatch.threads_h * plan.dispatch.threads_w);
}

TEST(
    GfxParallelism,
    SelectCapabilityEnabledDenseConvBacksChannelBlockDownToDivisibleLaneCount) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_channel_blocking_caps(), ov::Shape{1, 130, 80, 80}, 256, 130,
      256 * 3 * 3, false, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.output_channel_block, 2u);
  EXPECT_EQ(plan.dispatch.channel_block, 2u);
}

TEST(GfxParallelism,
     SelectBroadcomDepthwiseConvKeepsOneSpatialOutputPerThread) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_large_broadcom_caps(), ov::Shape{1, 64, 80, 80}, 1, 64, 3 * 3, false,
      true);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.dispatch.tile_h, plan.dispatch.threads_h);
  EXPECT_EQ(plan.dispatch.tile_w, plan.dispatch.threads_w);
  EXPECT_EQ(plan.dispatch.channel_block, 1u);
}

TEST(GfxParallelism,
     SelectBroadcomUnblockedConvKeepsOneSpatialOutputPerThread) {
  const auto plan = ov::gfx_plugin::select_conv_parallelism(
      make_large_broadcom_caps(), ov::Shape{1, 5, 80, 80}, 3, 5, 3 * 3 * 3,
      false, false);

  ASSERT_TRUE(plan.prefer_parallel);
  EXPECT_EQ(plan.dispatch.tile_h, plan.dispatch.threads_h);
  EXPECT_EQ(plan.dispatch.tile_w, plan.dispatch.threads_w);
  EXPECT_EQ(plan.dispatch.channel_block, 1u);
}

TEST(GfxParallelism,
     ParallelDispatchUsesRemainingBlockDimsAfterCanonicalizedConvLoop) {
  ov::gfx_plugin::ParallelDispatchConfig cfg;
  cfg.enabled = true;
  cfg.loop_dims = 4;
  cfg.tile_h = 8;
  cfg.tile_w = 8;
  cfg.threads_h = 8;
  cfg.threads_w = 8;

  const auto dispatch =
      ov::gfx_plugin::make_parallel_dispatch(ov::Shape{1, 1, 28, 28}, cfg);

  EXPECT_EQ(dispatch.grid[0], 32u);
  EXPECT_EQ(dispatch.grid[1], 32u);
  EXPECT_EQ(dispatch.grid[2], 1u);
  EXPECT_EQ(dispatch.threads_per_group[0], 8u);
  EXPECT_EQ(dispatch.threads_per_group[1], 8u);
  EXPECT_EQ(dispatch.threads_per_group[2], 1u);
}

TEST(GfxParallelism,
     ParallelDispatchKeepsThreadAxesWithCanonicalizedAsymmetricConvLoop) {
  ov::gfx_plugin::ParallelDispatchConfig cfg;
  cfg.enabled = true;
  cfg.loop_dims = 4;
  cfg.tile_h = 8;
  cfg.tile_w = 16;
  cfg.threads_h = 8;
  cfg.threads_w = 16;
  cfg.channel_block = 4;

  const auto dispatch =
      ov::gfx_plugin::make_parallel_dispatch(ov::Shape{1, 4, 80, 80}, cfg);

  EXPECT_EQ(dispatch.grid[0], 10u * 16u);
  EXPECT_EQ(dispatch.grid[1], 5u * 8u);
  EXPECT_EQ(dispatch.grid[2], 1u);
  EXPECT_EQ(dispatch.threads_per_group[0], 16u);
  EXPECT_EQ(dispatch.threads_per_group[1], 8u);
  EXPECT_EQ(dispatch.threads_per_group[2], 1u);
}

TEST(GfxParallelism, ParallelDispatchUsesChannelBlocksForConvLoop) {
  ov::gfx_plugin::ParallelDispatchConfig cfg;
  cfg.enabled = true;
  cfg.loop_dims = 5;
  cfg.tile_h = 8;
  cfg.tile_w = 8;
  cfg.threads_h = 8;
  cfg.threads_w = 8;
  cfg.channel_block = 2;

  const auto dispatch =
      ov::gfx_plugin::make_parallel_dispatch(ov::Shape{1, 64, 28, 28}, cfg);

  EXPECT_EQ(dispatch.grid[0], 32u * 8u);
  EXPECT_EQ(dispatch.grid[1], 4u * 8u);
  EXPECT_EQ(dispatch.grid[2], 4u);
}

TEST(GfxParallelism, RememberMatMulParallelismAllowsSerialFallbackOverride) {
  const auto caps = make_caps();
  const ov::Shape output_shape{1, 256, 256};

  const auto initial =
      ov::gfx_plugin::select_matmul_parallelism(caps, output_shape);
  ASSERT_TRUE(initial.prefer_parallel);

  ov::gfx_plugin::MatMulParallelismPlan serial_plan;
  serial_plan.prefer_parallel = false;
  serial_plan.variant = "serial";

  ov::gfx_plugin::remember_matmul_parallelism(caps, output_shape, serial_plan);

  const auto overridden =
      ov::gfx_plugin::select_matmul_parallelism(caps, output_shape);
  EXPECT_FALSE(overridden.prefer_parallel);
  EXPECT_EQ(overridden.variant, "serial");
  EXPECT_FALSE(overridden.dispatch.enabled);
}

TEST(GfxParallelism,
     SelectChunkDispatchPlanUsesLargeSingleDispatchForMidSizeOpenClWorkloads) {
  const auto plan = ov::gfx_plugin::select_chunk_dispatch_plan(
      make_caps(), "conv2d", 8192, 576);

  EXPECT_EQ(plan.threads_per_group, 64u);
  EXPECT_EQ(plan.elems_per_dispatch, 8192u);
  EXPECT_EQ(plan.variant, "conv2d_chunk_8192");
}

TEST(GfxParallelism,
     SelectChunkDispatchPlanCapsLargeOpenClWorkloadsToFewDispatches) {
  const auto plan = ov::gfx_plugin::select_chunk_dispatch_plan(
      make_caps(), "conv2d", 131072, 1152);

  EXPECT_EQ(plan.threads_per_group, 64u);
  EXPECT_EQ(plan.elems_per_dispatch, 16384u);
  EXPECT_EQ(plan.variant, "conv2d_chunk_16384");
}
