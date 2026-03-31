// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include <utility>

#include "gtest/gtest.h"

#include "runtime/gfx_parallelism.hpp"

namespace {

class TestDeviceInfoBufferManager final : public ov::gfx_plugin::GpuBufferManager {
public:
    explicit TestDeviceInfoBufferManager(ov::gfx_plugin::GpuExecutionDeviceInfo info) : m_info(std::move(info)) {}

    std::optional<ov::gfx_plugin::GpuExecutionDeviceInfo> query_execution_device_info() const override {
        return m_info;
    }

private:
    ov::gfx_plugin::GpuExecutionDeviceInfo m_info;
};

ov::gfx_plugin::GfxParallelismCaps make_caps() {
    ov::gfx_plugin::GfxParallelismCaps caps;
    caps.backend = ov::gfx_plugin::GpuBackend::Vulkan;
    caps.device_key = "test:vulkan";
    caps.preferred_simd_width = 32;
    caps.subgroup_size = 32;
    caps.max_total_threads_per_group = 64;
    caps.max_threads_per_group = {64, 64, 1};
    return caps;
}

}  // namespace

TEST(GfxParallelism, SelectMatMulParallelismPrefersSquareTilesForSquareOutputs) {
    const auto plan = ov::gfx_plugin::select_matmul_parallelism(make_caps(), ov::Shape{1, 256, 256});

    ASSERT_TRUE(plan.prefer_parallel);
    EXPECT_EQ(plan.dispatch.threads_h, 8u);
    EXPECT_EQ(plan.dispatch.threads_w, 8u);
}

TEST(GfxParallelism, QueryParallelismCapsUsesBackendNeutralDeviceInfo) {
    ov::gfx_plugin::GpuExecutionDeviceInfo info;
    info.backend = ov::gfx_plugin::GpuBackend::Vulkan;
    info.device_key = "test-device";
    info.preferred_simd_width = 16;
    info.subgroup_size = 32;
    info.max_total_threads_per_group = 128;
    info.max_threads_per_group = {128, 32, 8};

    TestDeviceInfoBufferManager buffer_manager(info);
    const auto caps = ov::gfx_plugin::query_parallelism_caps(&buffer_manager);

    EXPECT_EQ(caps.backend, ov::gfx_plugin::GpuBackend::Vulkan);
    EXPECT_EQ(caps.device_key, "test-device");
    EXPECT_EQ(caps.preferred_simd_width, 16u);
    EXPECT_EQ(caps.subgroup_size, 32u);
    EXPECT_EQ(caps.max_total_threads_per_group, 128u);
    EXPECT_EQ(caps.max_threads_per_group[0], 128u);
    EXPECT_EQ(caps.max_threads_per_group[1], 32u);
    EXPECT_EQ(caps.max_threads_per_group[2], 8u);
}

TEST(GfxParallelism, SelectMatMulParallelismPrefersWideTilesForWideOutputs) {
    const auto plan = ov::gfx_plugin::select_matmul_parallelism(make_caps(), ov::Shape{128, 1600});

    ASSERT_TRUE(plan.prefer_parallel);
    EXPECT_TRUE(plan.dispatch.enabled);
    EXPECT_GT(plan.dispatch.threads_h, 0u);
    EXPECT_GT(plan.dispatch.threads_w, 0u);
}

TEST(GfxParallelism, RememberMatMulParallelismAllowsSerialFallbackOverride) {
    const auto caps = make_caps();
    const ov::Shape output_shape{1, 256, 256};

    const auto initial = ov::gfx_plugin::select_matmul_parallelism(caps, output_shape);
    ASSERT_TRUE(initial.prefer_parallel);

    ov::gfx_plugin::MatMulParallelismPlan serial_plan;
    serial_plan.prefer_parallel = false;
    serial_plan.variant = "serial";

    ov::gfx_plugin::remember_matmul_parallelism(caps, output_shape, serial_plan);

    const auto overridden = ov::gfx_plugin::select_matmul_parallelism(caps, output_shape);
    EXPECT_FALSE(overridden.prefer_parallel);
    EXPECT_EQ(overridden.variant, "serial");
    EXPECT_FALSE(overridden.dispatch.enabled);
}

TEST(GfxParallelism, SelectChunkDispatchPlanUsesLargeSingleDispatchForMidSizeVulkanWorkloads) {
    const auto plan = ov::gfx_plugin::select_chunk_dispatch_plan(make_caps(), "conv2d", 8192, 576);

    EXPECT_EQ(plan.threads_per_group, 64u);
    EXPECT_EQ(plan.elems_per_dispatch, 8192u);
    EXPECT_EQ(plan.variant, "conv2d_chunk_8192");
}

TEST(GfxParallelism, SelectChunkDispatchPlanCapsLargeVulkanWorkloadsToFewDispatches) {
    const auto plan = ov::gfx_plugin::select_chunk_dispatch_plan(make_caps(), "conv2d", 131072, 1152);

    EXPECT_EQ(plan.threads_per_group, 64u);
    EXPECT_EQ(plan.elems_per_dispatch, 16384u);
    EXPECT_EQ(plan.variant, "conv2d_chunk_16384");
}
