// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "runtime/gfx_parallelism.hpp"

namespace {

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

TEST(GfxParallelism, SelectMatMulParallelismPrefersWideTilesForWideOutputs) {
    const auto plan = ov::gfx_plugin::select_matmul_parallelism(make_caps(), ov::Shape{128, 1600});

    ASSERT_TRUE(plan.prefer_parallel);
    EXPECT_TRUE(plan.dispatch.enabled);
    EXPECT_GT(plan.dispatch.threads_h, 0u);
    EXPECT_GT(plan.dispatch.threads_w, 0u);
}
