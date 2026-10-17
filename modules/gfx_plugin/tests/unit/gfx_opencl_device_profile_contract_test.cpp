// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "backends/opencl/runtime/opencl_api.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

class OpenClDeviceSelectionContract final {
public:
    static void apply_dispatch_limits(OpenClDeviceSelection& selection) {
        selection.compute_units = 8;
        selection.max_work_group_size = 256;
        selection.max_work_item_sizes = {256, 256, 64};
    }

    static OpenClDeviceSelection raspberry_pi_clvk_v3d() {
        OpenClDeviceSelection selection;
        selection.device_type = CL_DEVICE_TYPE_GPU;
        selection.platform_name = "clvk";
        selection.device_name = "V3D 7.1.7.0";
        selection.vendor_name = "Unknown vendor";
        selection.device_version = "OpenCL 3.0 CLVK on Vulkan";
        selection.driver_version = "3.0 CLVK on Vulkan";
        apply_dispatch_limits(selection);
        return selection;
    }

    static OpenClDeviceSelection android_adreno() {
        OpenClDeviceSelection selection;
        selection.device_type = CL_DEVICE_TYPE_GPU;
        selection.platform_name = "Qualcomm OpenCL";
        selection.device_name = "Adreno 740";
        selection.vendor_name = "Qualcomm";
        selection.device_version = "OpenCL 3.0";
        selection.driver_version = "Qualcomm OpenCL";
        apply_dispatch_limits(selection);
        return selection;
    }

    static OpenClDeviceSelection raspberry_pi_rusticl_v3d() {
        auto selection = raspberry_pi_clvk_v3d();
        selection.platform_name = "rusticl";
        selection.vendor_name = "Mesa";
        selection.device_version = "OpenCL 3.0 Mesa Rusticl";
        selection.driver_version = "Mesa Rusticl";
        return selection;
    }

    static OpenClDeviceSelection raspberry_pi_non_clvk_v3d() {
        auto selection = raspberry_pi_clvk_v3d();
        selection.platform_name = "Mesa";
        selection.vendor_name = "Broadcom";
        selection.device_version = "OpenCL 3.0";
        selection.driver_version = "Mesa";
        return selection;
    }
};

TEST(GfxOpenClDeviceProfileContractTest,
     RaspberryPiV3dUsesClvkAsProductionRoute) {
    EXPECT_NO_THROW(validate_opencl_device_selection(
        OpenClDeviceSelectionContract::raspberry_pi_clvk_v3d()));
}

TEST(GfxOpenClDeviceProfileContractTest,
     RaspberryPiV3dRejectsRusticlAsRuntimeAcceptanceRoute) {
    EXPECT_THROW(validate_opencl_device_selection(
                     OpenClDeviceSelectionContract::raspberry_pi_rusticl_v3d()),
                 ov::Exception);
}

TEST(GfxOpenClDeviceProfileContractTest,
     RaspberryPiV3dRequiresClvkProfileIdentity) {
    EXPECT_THROW(validate_opencl_device_selection(
                     OpenClDeviceSelectionContract::raspberry_pi_non_clvk_v3d()),
                 ov::Exception);
}

TEST(GfxOpenClDeviceProfileContractTest,
     AndroidAdrenoExecutionInfoUsesOpenClAdrenoProfile) {
    const auto info = make_opencl_execution_device_info(
        OpenClDeviceSelectionContract::android_adreno());

    EXPECT_EQ(info.device_family, GpuDeviceFamily::QualcommAdreno);
    EXPECT_EQ(info.parallelism_profile.profile_key, "opencl:adreno");
    EXPECT_EQ(info.preferred_simd_width, 32u);
    EXPECT_EQ(info.subgroup_size, 32u);
    EXPECT_EQ(info.max_total_threads_per_group, 256u);
    EXPECT_TRUE(info.supports_conv_output_channel_blocking);
    EXPECT_TRUE(info.supports_conv_channel_block_spatial_tiling);
    EXPECT_TRUE(info.parallelism_profile.supports_conv_output_channel_blocking);
    EXPECT_TRUE(
        info.parallelism_profile.supports_conv_channel_block_spatial_tiling);
}

TEST(GfxOpenClDeviceProfileContractTest,
     RaspberryPiV3dExecutionInfoUsesOpenClBroadcomV3dProfile) {
    const auto info = make_opencl_execution_device_info(
        OpenClDeviceSelectionContract::raspberry_pi_clvk_v3d());

    EXPECT_EQ(info.device_family, GpuDeviceFamily::BroadcomV3D);
    EXPECT_EQ(info.parallelism_profile.profile_key, "opencl:broadcom_v3d");
    EXPECT_EQ(info.preferred_simd_width, 16u);
    EXPECT_EQ(info.subgroup_size, 16u);
    EXPECT_EQ(info.max_total_threads_per_group, 64u);
    EXPECT_EQ(info.max_threads_per_group[0], 64u);
    EXPECT_EQ(info.max_threads_per_group[1], 64u);
    EXPECT_EQ(info.max_threads_per_group[2], 16u);
    EXPECT_FALSE(info.supports_conv_output_channel_blocking);
    EXPECT_FALSE(info.supports_conv_channel_block_spatial_tiling);
    EXPECT_TRUE(info.parallelism_profile.enable_skinny_matmul_tiles);
    EXPECT_TRUE(
        info.parallelism_profile.chunk_dispatch.retune_threads_to_workload);
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
