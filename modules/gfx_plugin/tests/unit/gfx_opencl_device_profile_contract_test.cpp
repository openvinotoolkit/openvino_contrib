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
    static OpenClDeviceSelection raspberry_pi_clvk_v3d() {
        OpenClDeviceSelection selection;
        selection.device_type = CL_DEVICE_TYPE_GPU;
        selection.platform_name = "clvk";
        selection.device_name = "V3D 7.1.7.0";
        selection.vendor_name = "Unknown vendor";
        selection.device_version = "OpenCL 3.0 CLVK on Vulkan";
        selection.driver_version = "3.0 CLVK on Vulkan";
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

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
