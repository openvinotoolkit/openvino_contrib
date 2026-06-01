// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "compiler/backend_config.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

void expect_opencl_profile_contract_unavailable() {
    EXPECT_FALSE(kGfxBackendOpenCLAvailable)
        << "This adapter is linked only when the OpenCL backend module is not "
           "present in the current build target.";
}

TEST(GfxOpenClDeviceProfileContractTest,
     RaspberryPiV3dUsesClvkAsProductionRoute) {
    expect_opencl_profile_contract_unavailable();
}

TEST(GfxOpenClDeviceProfileContractTest,
     RaspberryPiV3dRejectsRusticlAsRuntimeAcceptanceRoute) {
    expect_opencl_profile_contract_unavailable();
}

TEST(GfxOpenClDeviceProfileContractTest,
     RaspberryPiV3dRequiresClvkProfileIdentity) {
    expect_opencl_profile_contract_unavailable();
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
